import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import Line
from itertools import combinations
from court_reference import CourtReference
from skimage.measure import label
import scipy.signal as sp


class CourtDetector:
    """
    Detecting and tracking court in frame
    """

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.colour_threshold = 200
        self.dist_tau = 3
        self.intensity_threshold = 40
        self.court_reference = CourtReference()
        self.v_width = 0
        self.v_height = 0
        self.frame = None
        self.gray = None
        self.court_warp_matrix = []
        self.game_warp_matrix = []
        self.court_score = 0
        self.baseline_top = None
        self.baseline_bottom = None
        self.net = None
        self.left_court_line = None
        self.right_court_line = None
        self.left_inner_line = None
        self.right_inner_line = None
        self.middle_line_top = None
        self.middle_line_bottom = None
        self.top_inner_line = None
        self.bottom_inner_line = None
        self.top_second_inner_line = None
        self.bottom_second_inner_line = None
        self.success_flag = False
        self.success_accuracy = 80
        self.success_score = 1000
        self.best_conf = None
        self.frame_points = None
        self.dist = 5
        self.camera_position = 0

    def detect(self, frame, verbose=0, position=0):
        """
        Detecting the court in the frame
        """
        self.verbose = verbose
        self.frame = frame
        self.v_height, self.v_width = frame.shape[:2]
        # Get binary image from the frame
        self.gray = self._threshold(frame)
        self.camera_position = position

        # Filter pixel using the court known structure
        filtered = self._filter_pixels(self.gray)

        # Detect lines using Hough transform
        horizontal_lines, vertical_lines = self._detect_lines(filtered)

        # Find transformation from reference court to frame`s court
        court_warp_matrix, game_warp_matrix, self.court_score = self._find_homography(horizontal_lines,
                                                                                      vertical_lines)
        self.court_warp_matrix.append(court_warp_matrix)
        self.game_warp_matrix.append(game_warp_matrix)
        court_accuracy = self._get_court_accuracy()
        if court_accuracy > self.success_accuracy and self.court_score > self.success_score:
            self.success_flag = True
        print('Court accuracy = %.2f' % court_accuracy)
        # Find important lines location on frame
        self.find_lines_location()

        '''game_warped = cv2.warpPerspective(self.frame, self.game_warp_matrix,
                                          (self.court_reference.court.shape[1], self.court_reference.court.shape[0]))
        cv2.imwrite('../result/img7.png', game_warped)'''

    def _threshold(self, frame):
        """
        Simple thresholding for white pixels
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        index = 0
        for i in range(0, 255, 1):
            index = 255 - i
            gray_precent = sum(hist[index:255]) / (self.v_width * self.v_height)
            if gray_precent >= 0.05:
                break
        self.colour_threshold = index
        gray = cv2.threshold(gray, self.colour_threshold, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow('gray2', gray)
        return gray

    def _filter_pixels(self, gray):
        """
        Filter pixels by using the court line structure
        """
        for i in range(self.dist_tau, len(gray) - self.dist_tau):
            for j in range(self.dist_tau, len(gray[0]) - self.dist_tau):
                if gray[i, j] == 0:
                    continue
                if (gray[i, j] - gray[i + self.dist_tau, j] > self.intensity_threshold and
                        gray[i, j] - gray[i - self.dist_tau, j] > self.intensity_threshold):
                    continue

                if (gray[i, j] - gray[i, j + self.dist_tau] > self.intensity_threshold and
                        gray[i, j] - gray[i, j - self.dist_tau] > self.intensity_threshold):
                    continue
                gray[i, j] = 0
        return gray

    def _detect_lines(self, gray):
        """
        Finds all line in frame using Hough transform
        """
        minLineLength = int(self.v_width * 0.15)
        maxLineGap = 20
        # Detect all lines
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
        lines = np.squeeze(lines)
        if self.verbose:
            display_lines_on_frame(self.frame.copy(), [], lines)

        # Classify the lines using their slope
        horizontal, vertical = self._classify_lines(lines)
        if self.verbose:
            display_lines_on_frame(self.frame.copy(), horizontal, vertical)

        # Merge lines that belong to the same line on frame
        horizontal, vertical = self._merge_lines(horizontal, vertical)
        if self.verbose:
            display_lines_on_frame(self.frame.copy(), horizontal, vertical)

        return horizontal, vertical

    def _classify_lines(self, lines):
        """
        Classify line to vertical and horizontal lines
        """
        horizontal = []
        vertical = []
        highest_vertical_y = np.inf
        lowest_vertical_y = 0
        if self.camera_position == 0:
            for line in lines:
                x1, y1, x2, y2 = line
                dx = abs(x1 - x2)
                dy = abs(y1 - y2)
                if dx > 2 * dy:
                    horizontal.append(line)
                else:
                    vertical.append(line)
                    highest_vertical_y = min(highest_vertical_y, y1, y2)
                    lowest_vertical_y = max(lowest_vertical_y, y1, y2)

        if self.camera_position == 1:
            for line in lines:
                x1, y1, x2, y2 = line
                if y2 - y1 > 5:
                    horizontal.append(line)
                elif y2 - y1 < -5:
                    vertical.append(line)
                    highest_vertical_y = min(highest_vertical_y, y1, y2)
                    lowest_vertical_y = max(lowest_vertical_y, y1, y2)

        if self.camera_position == 2:
            for line in lines:
                x1, y1, x2, y2 = line
                if y2 - y1 < -5:
                    horizontal.append(line)
                elif y2 - y1 > 5:
                    vertical.append(line)
                    highest_vertical_y = min(highest_vertical_y, y1, y2)
                    lowest_vertical_y = max(lowest_vertical_y, y1, y2)

        # Filter horizontal lines using vertical lines lowest and highest point
        clean_horizontal = []
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        highest_vertical_y -= h * 2 / 15
        for line in horizontal:
            x1, y1, x2, y2 = line
            if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                clean_horizontal.append(line)

        return clean_horizontal, vertical

    def _classify_vertical(self, vertical, width):
        """
        Classify vertical lines to right and left vertical lines using the location on frame
        """
        vertical_lines = []
        vertical_left = []
        vertical_right = []
        right_th = width * 4 / 7
        left_th = width * 3 / 7
        for line in vertical:
            x1, y1, x2, y2 = line
            if x1 < left_th or x2 < left_th:
                vertical_left.append(line)
            elif x1 > right_th or x2 > right_th:
                vertical_right.append(line)
            else:
                vertical_lines.append(line)
        return vertical_lines, vertical_left, vertical_right

    def _merge_lines(self, horizontal_lines, vertical_lines):
        """
        Merge lines that belongs to the same frame`s lines
        """
        """
        # Merge horizontal lines
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dy = abs(y3 - y2)
                        if dy < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_horizontal_lines.append(line)
        """
        if self.camera_position == 0:
            horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
            mask = [True] * len(horizontal_lines)
            new_horizontal_lines = []
            for i, line in enumerate(horizontal_lines):
                if mask[i]:
                    for j, s_line in enumerate(horizontal_lines[i + 1:]):
                        if mask[i + j + 1]:
                            x1, y1, x2, y2 = line
                            x3, y3, x4, y4 = s_line
                            dy = abs(y3 - y2)
                            if dy < 10:
                                points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                                line = np.array([*points[0], *points[-1]])
                                mask[i + j + 1] = False
                    new_horizontal_lines.append(line)
        else:
            # Merge horizontal lines
            horizontal_lines = sorted(horizontal_lines, key=lambda item: item[1])
            xl, yl, xr, yr = (0, self.v_height * 6 / 7, self.v_width, self.v_height * 6 / 7)
            mask = [True] * len(horizontal_lines)
            new_horizontal_lines = []
            for i, line in enumerate(horizontal_lines):
                if mask[i]:
                    for j, s_line in enumerate(horizontal_lines[i + 1:]):
                        if mask[i + j + 1]:
                            x1, y1, x2, y2 = line
                            x3, y3, x4, y4 = s_line
                            xi, yi = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                            xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))
                            dy = abs(xi - xj)
                            if dy < 10:
                                points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                                line = np.array([*points[0], *points[-1]])
                                mask[i + j + 1] = False
                    new_horizontal_lines.append(line)

        # Merge vertical lines
        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        xl, yl, xr, yr = (0, self.v_height * 6 / 7, self.v_width, self.v_height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                        xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))

                        dx = abs(xi - xj)
                        if dx < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False

                new_vertical_lines.append(line)
        return new_horizontal_lines, new_vertical_lines

    def _find_homography(self, horizontal_lines, vertical_lines):
        """
        Finds transformation from reference court to frame`s court using 4 pairs of matching points
        """
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None
        k = 0
        # Loop over every pair of horizontal lines and every pair of vertical lines
        for horizontal_pair in list(combinations(horizontal_lines, 2)):
            for vertical_pair in list(combinations(vertical_lines, 2)):
                h1, h2 = horizontal_pair
                v1, v2 = vertical_pair
                # Finding intersection points of all lines
                i1 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i2 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
                i3 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i4 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[0:2]), tuple(v2[2:])))

                intersections = [i1, i2, i3, i4]
                intersections = sort_intersection_points(intersections)

                for i, configuration in self.court_reference.court_conf.items():
                    # Find transformation
                    matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(intersections), method=0)
                    inv_matrix = cv2.invert(matrix)[1]
                    # Get transformation score
                    confi_score = self._get_confi_score(matrix)

                    if max_score < confi_score:
                        max_score = confi_score
                        max_mat = matrix
                        max_inv_mat = inv_matrix
                        self.best_conf = i

                    k += 1

        if self.verbose:
            # if self.verbose:
            frame = self.frame.copy()
            court = self.add_court_overlay(frame, max_mat, (255, 0, 0))
            cv2.imshow('court', court)
            # cv2.imwrite('../result/img6.png', court)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()
        print(f'Score = {max_score}')
        print(f'Combinations tested = {k}')

        return max_mat, max_inv_mat, max_score

    def _get_confi_score(self, matrix):
        """
        Calculate transformation score
        """
        court = cv2.warpPerspective(self.court_reference.court, matrix, self.frame.shape[1::-1])
        court[court > 0] = 1
        gray = self.gray.copy()
        gray[gray > 0] = 1
        correct = court * gray
        wrong = court - correct
        c_p = np.sum(correct)
        w_p = np.sum(wrong)
        return c_p - 0.5 * w_p

    def add_court_overlay(self, frame, homography=None, overlay_color=(255, 255, 255), frame_num=-1):
        """
        Add overlay of the court to the frame
        """
        if homography is None and len(self.court_warp_matrix) > 0 and frame_num < len(self.court_warp_matrix):
            homography = self.court_warp_matrix[frame_num]
        court = cv2.warpPerspective(self.court_reference.court, homography, frame.shape[1::-1])
        frame[court > 0, :] = overlay_color
        return frame

    def find_lines_location(self):
        """
        Finds important lines location on frame
        """
        p = np.array(self.court_reference.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
        lines = cv2.perspectiveTransform(p, self.court_warp_matrix[-1]).reshape(-1)
        self.baseline_top = lines[:4]
        self.baseline_bottom = lines[4:8]
        self.net = lines[8:12]
        self.left_court_line = lines[12:16]
        self.right_court_line = lines[16:20]
        self.left_inner_line = lines[20:24]
        self.right_inner_line = lines[24:28]
        self.middle_line_top = lines[28:32]
        self.middle_line_bottom = lines[32:36]
        self.top_inner_line = lines[36:40]
        self.bottom_inner_line = lines[40:44]
        self.top_second_inner_line = lines[44:48]
        self.bottom_second_inner_line = lines[48:52]
        if self.verbose:
            display_lines_on_frame(self.frame.copy(), [self.baseline_top, self.baseline_bottom,
                                                       self.net, self.top_inner_line, self.bottom_inner_line,
                                                       self.top_second_inner_line, self.bottom_second_inner_line],
                                   [self.left_court_line, self.right_court_line,
                                    self.right_inner_line, self.left_inner_line, self.middle_line_top,
                                    self.middle_line_bottom])

    def get_extra_parts_location(self, frame_num=-1):
        parts = np.array(self.court_reference.get_extra_parts(), dtype=np.float32).reshape((-1, 1, 2))
        parts = cv2.perspectiveTransform(parts, self.court_warp_matrix[frame_num]).reshape(-1)
        top_part = parts[:2]
        bottom_part = parts[2:]
        return top_part, bottom_part

    def delete_extra_parts(self, frame, frame_num=-1):
        img = frame.copy()
        top, bottom = self.get_extra_parts_location(frame_num)
        img[int(bottom[1] - 10):int(bottom[1] + 10), int(bottom[0] - 15):int(bottom[0] + 15), :] = (0, 0, 0)
        img[int(top[1] - 10):int(top[1] + 10), int(top[0] - 15):int(top[0] + 15), :] = (0, 0, 0)
        return img

    def get_warped_court(self):
        """
        Returns warped court using the reference court and the transformation of the court
        """
        court = cv2.warpPerspective(self.court_reference.court, self.court_warp_matrix[-1], self.frame.shape[1::-1])
        court[court > 0] = 1
        return court

    def _get_court_accuracy(self, verbose=0):
        """
        Calculate court accuracy after detection
        """
        frame = self.frame.copy()
        gray = self._threshold(frame)
        gray[gray > 0] = 1
        gray = cv2.dilate(gray, np.ones((9, 9), dtype=np.uint8))
        court = self.get_warped_court()
        total_white_pixels = sum(sum(court))
        sub = court.copy()
        sub[gray == 1] = 0
        accuracy = 100 - (sum(sum(sub)) / total_white_pixels) * 100
        if verbose:
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(gray, cmap='gray')
            plt.title('Grayscale frame'), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 3, 2)
            plt.imshow(court, cmap='gray')
            plt.title('Projected court'), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 3, 3)
            plt.imshow(sub, cmap='gray')
            plt.title('Subtraction result'), plt.xticks([]), plt.yticks([])
            plt.show()
        return accuracy

    def track_court(self, frame):
        """
        Track court location after detection
        """
        copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.frame_points is None:
            conf_points = np.array(self.court_reference.court_conf[self.best_conf], dtype=np.float32).reshape(
                (-1, 1, 2))
            self.frame_points = cv2.perspectiveTransform(conf_points,
                                                         self.court_warp_matrix[-1]).squeeze().round()
        # Lines of configuration on frames
        line1 = self.frame_points[:2]
        line2 = self.frame_points[2:4]
        line3 = self.frame_points[[0, 2]]
        line4 = self.frame_points[[1, 3]]
        lines = [line1, line2, line3, line4]
        new_lines = []
        for line in lines:
            # Get 100 samples of each line in the frame
            points_on_line = np.linspace(line[0], line[1], 102)[1:-1]  # 100 samples on the line
            p1 = None
            p2 = None
            if line[0][0] > self.v_width or line[0][0] < 0 or line[0][1] > self.v_height or line[0][1] < 0:
                for p in points_on_line:
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p1 = p
                        break
            if line[1][0] > self.v_width or line[1][0] < 0 or line[1][1] > self.v_height or line[1][1] < 0:
                for p in reversed(points_on_line):
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p2 = p
                        break
            # if one of the ends of the line is out of the frame get only the points inside the frame
            if p1 is not None or p2 is not None:
                print('points outside screen')
                points_on_line = np.linspace(p1 if p1 is not None else line[0], p2 if p2 is not None else line[1], 102)[
                                 1:-1]

            new_points = []
            # Find max intensity pixel near each point
            for p in points_on_line:
                p = (int(round(p[0])), int(round(p[1])))
                top_y, top_x = max(p[1] - self.dist, 0), max(p[0] - self.dist, 0)
                bottom_y, bottom_x = min(p[1] + self.dist, self.v_height), min(p[0] + self.dist, self.v_width)
                patch = gray[top_y: bottom_y, top_x: bottom_x]
                y, x = np.unravel_index(np.argmax(patch), patch.shape)
                if patch[y, x] > 150:
                    new_p = (x + top_x + 1, y + top_y + 1)
                    new_points.append(new_p)
                    cv2.circle(copy, p, 1, (255, 0, 0), 1)
                    cv2.circle(copy, new_p, 1, (0, 0, 255), 1)
            new_points = np.array(new_points, dtype=np.float32).reshape((-1, 1, 2))
            # find line fitting the new points
            [vx, vy, x, y] = cv2.fitLine(new_points, cv2.DIST_L2, 0, 0.01, 0.01)
            new_lines.append(((int(x - vx * self.v_width), int(y - vy * self.v_width)),
                              (int(x + vx * self.v_width), int(y + vy * self.v_width))))

            # if less than 50 points were found detect court from the start instead of tracking
            if len(new_points) < 50:
                if self.dist > 20:
                    cv2.imshow('court', copy)
                    if cv2.waitKey(0) & 0xff == 27:
                        cv2.destroyAllWindows()
                    self.detect(frame)
                    conf_points = np.array(self.court_reference.court_conf[self.best_conf], dtype=np.float32).reshape(
                        (-1, 1, 2))
                    self.frame_points = cv2.perspectiveTransform(conf_points,
                                                                 self.court_warp_matrix[-1]).squeeze().round()

                    print('Smaller than 50')
                    return
                else:
                    print('Court tracking failed, adding 5 pixels to dist')
                    self.dist += 5
                    self.track_court(frame)
                    return
        # Find transformation from new lines
        i1 = line_intersection(new_lines[0], new_lines[2])
        i2 = line_intersection(new_lines[0], new_lines[3])
        i3 = line_intersection(new_lines[1], new_lines[2])
        i4 = line_intersection(new_lines[1], new_lines[3])
        intersections = np.array([i1, i2, i3, i4], dtype=np.float32)
        matrix, _ = cv2.findHomography(np.float32(self.court_reference.court_conf[self.best_conf]),
                                       intersections, method=0)
        inv_matrix = cv2.invert(matrix)[1]
        self.court_warp_matrix.append(matrix)
        self.game_warp_matrix.append(inv_matrix)
        self.frame_points = intersections


def largestConnectComponent(bw_img, ):
    labeled_img, num = label(bw_img, background=0, return_num=True)
    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 0
    max_num = 0
    # 这里从1开始，防止将背景设置为最大连通域
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    labeled_img[np.where(labeled_img == max_label)] = 255
    labeled_img[np.where(labeled_img != 255)] = 0

    return labeled_img


def sort_intersection_points(intersections):
    """
    sort intersection points from top left to bottom right
    """
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34


def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates


def display_lines_on_frame(frame, horizontal=(), vertical=()):
    """
    Display lines on frame for horizontal and vertical lines
    """

    '''cv2.line(frame, (int(len(frame[0]) * 4 / 7), 0), (int(len(frame[0]) * 4 / 7), 719), (255, 255, 0), 2)
    cv2.line(frame, (int(len(frame[0]) * 3 / 7), 0), (int(len(frame[0]) * 3 / 7), 719), (255, 255, 0), 2)'''
    for line in horizontal:
        x1, y1, x2, y2 = line
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    for line in vertical:
        x1, y1, x2, y2 = line
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    cv2.imshow('court', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return frame


def display_lines_and_points_on_frame(frame, lines=(), points=(), line_color=(0, 0, 255), point_color=(255, 0, 0)):
    """
    Display all lines and points given on frame
    """

    for line in lines:
        x1, y1, x2, y2 = line
        frame = cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
    for p in points:
        frame = cv2.circle(frame, p, 2, point_color, 2)

    cv2.imshow('court', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return frame


def get_target_img_area(frame):
    # 1. 2g-r-b提取绿色
    fsrc = np.array(img, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = 2 * g - b - r

    # 1.1 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # 1.2 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])

    # 1.3 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)

    # 2. 图像膨胀
    kernel = np.ones((5, 5), np.uint8)
    dilate_img = cv2.dilate(bin_img, kernel, iterations=2)

    # 3. 得到最大绿色区域
    label_img = np.uint8(largestConnectComponent(dilate_img))

    # 1.4 得到彩色的图像
    (b8, g8, r8) = cv2.split(img)
    target_img = cv2.merge([b8 & label_img, g8 & label_img, r8 & label_img])
    return target_img


if __name__ == '__main__':
    filename = '../images/img4.jpg'
    img = cv2.imread(filename)
    import time

    s = time.time()
    color_img = get_target_img_area(img)
    court_detector = CourtDetector()
    court_detector.detect(color_img, 0, 0)
    top, bottom = court_detector.get_extra_parts_location()
    cv2.circle(img, (int(top[0]), int(top[1])), 30, (0, 255, 0), 1)
    cv2.circle(img, (int(bottom[0]), int(bottom[1])), 30, (0, 255, 0), 1)
    img[int(bottom[1] - 10):int(bottom[1] + 10), int(bottom[0] - 10):int(bottom[0] + 10), :] = (255, 0, 0)
    img[int(top[1] - 10):int(top[1] + 10), int(top[0] - 10):int(top[0] + 10), :] = (255, 0, 0)
    cv2.imshow('df', img)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
    print(f'time = {time.time() - s}')
