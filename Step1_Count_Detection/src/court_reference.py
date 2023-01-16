import cv2
import numpy as np
import matplotlib.pyplot as plt


class CourtReference:
    """
    Court reference model
    """
    def __init__(self):
        """
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))
        self.top_extra_part = (832.5, 580)
        self.bottom_extra_part = (832.5, 2910)
        """
        self.baseline_top = ((202, 202), (808, 202))
        self.baseline_bottom = ((202, 1538), (808, 1538))
        self.net = ((202, 870), (808, 870))
        self.left_court_line = ((202, 202), (202, 1538))
        self.right_court_line = ((808, 202), (808, 1538))
        self.left_inner_line = ((248, 202), (248, 1538))
        self.right_inner_line = ((762, 202), (762, 1538))
        self.middle_line = ((505, 202), (505, 1538))
        self.middle_line_top = ((505, 202), (505, 670))
        self.middle_line_bottom = ((505, 1070), (505, 1538))
        self.top_inner_line = ((202, 278), (808, 278))
        self.top_second_inner_line = ((202, 670), (808, 670))
        self.bottom_inner_line = ((202, 1462), (808, 1462))
        self.bottom_second_inner_line = ((202, 1070), (808, 1070))
        self.top_extra_part = (505, 202)
        self.bottom_extra_part = (505, 1538)

        self.court_conf = {1: [*self.baseline_top, *self.baseline_bottom],
                           2: [*self.baseline_top, *self.top_inner_line],
                           3: [*self.baseline_top, *self.top_second_inner_line],
                           4: [*self.baseline_top, *self.bottom_second_inner_line],
                           5: [*self.baseline_top, *self.bottom_inner_line],
                           6: [*self.top_inner_line, *self.top_second_inner_line],
                           7: [*self.top_inner_line, *self.bottom_second_inner_line],
                           8: [*self.top_inner_line, *self.bottom_inner_line],
                           9: [*self.top_inner_line, *self.baseline_bottom],
                           10: [*self.top_second_inner_line, *self.bottom_second_inner_line],
                           11: [*self.top_second_inner_line, *self.bottom_inner_line],
                           12: [*self.top_second_inner_line, *self.baseline_bottom],
                           13: [*self.bottom_second_inner_line, *self.bottom_inner_line],
                           14: [*self.bottom_second_inner_line, *self.baseline_bottom],
                           15: [*self.bottom_inner_line, *self.baseline_bottom],
                           16: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1],
                                self.right_inner_line[1]],
                           17: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[1],
                                self.left_inner_line[1]],
                           18: [self.left_court_line[0], self.middle_line[0], self.left_court_line[1],
                                self.middle_line[1]],
                           19: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1],
                                self.right_court_line[1]],
                           20: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1],
                                self.right_inner_line[1]],
                           21: [self.left_inner_line[0], self.middle_line[0], self.left_inner_line[1],
                                self.middle_line[1]],
                           22: [self.middle_line[0], self.right_court_line[0], self.middle_line[1],
                                self.right_court_line[1]],
                           23: [self.middle_line[0], self.right_inner_line[0], self.middle_line[1],
                                self.right_inner_line[1]],
                           24: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1],
                                self.right_court_line[1]],
                           25: [self.left_court_line[0], self.middle_line_top[0], self.middle_line_top[1], self.top_second_inner_line[0]],
                           26: [self.top_extra_part, self.right_court_line[0], self.top_second_inner_line[1], self.middle_line_top[1]],
                           27: [self.left_court_line[0], self.middle_line_top[0], self.middle_line_bottom[0], self.bottom_second_inner_line[0]],
                           28: [self.top_extra_part, self.right_court_line[0], self.bottom_second_inner_line[1], self.middle_line_bottom[0]],
                           29: [self.top_second_inner_line[0], self.middle_line_top[1], self.middle_line_bottom[0], self.bottom_second_inner_line[0]],
                           30: [self.middle_line_top[1], self.top_second_inner_line[1], self.bottom_second_inner_line[1], self.middle_line_bottom[0]],
                           31: [self.top_second_inner_line[0], self.middle_line_top[1], self.middle_line_bottom[1], self.baseline_bottom[0]],
                           32: [self.middle_line_top[1], self.top_second_inner_line[1], self.baseline_bottom[1], self.middle_line_bottom[1]],
                           33: [self.bottom_second_inner_line[0], self.middle_line_bottom[0], self.middle_line_bottom[1], self.baseline_bottom[0]],
                           34: [self.middle_line_bottom[0], self.bottom_second_inner_line[1], self.baseline_bottom[1], self.middle_line_bottom[1]],
                           35: [self.bottom_second_inner_line[0], self.middle_line_bottom[0], (505, 1462), self.bottom_inner_line[0]]}

        """
        self.court_conf = {1: [*self.baseline_top, *self.baseline_bottom],
                           2: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1],
                               self.right_inner_line[1]],
                           3: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1],
                               self.right_court_line[1]],
                           4: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1],
                               self.right_inner_line[1]],
                           5: [*self.top_inner_line, *self.bottom_inner_line],
                           6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
                           7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
                           8: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1],
                               self.right_court_line[1]],
                           9: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[1],
                               self.left_inner_line[1]],
                           10: [self.top_inner_line[0], self.middle_line[0], self.bottom_inner_line[0],
                                self.middle_line[1]],
                           11: [self.middle_line[0], self.top_inner_line[1], self.middle_line[1],
                                self.bottom_inner_line[1]],
                           12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]]}
        """
        self.line_width = 1
        self.court_width = 610
        self.court_height = 1340
        self.top_bottom_border = 200
        self.right_left_border = 200
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_height + self.top_bottom_border * 2

        self.court = cv2.cvtColor(cv2.imread('court_configurations/court_reference.png'), cv2.COLOR_BGR2GRAY)

    def build_court_reference(self):
        """
        Create court reference image using the lines positions
        """
        court = np.zeros((self.court_height + 2 * self.top_bottom_border, self.court_width + 2 * self.right_left_border), dtype=np.uint8)
        cv2.line(court, *self.baseline_top, 1, self.line_width)
        cv2.line(court, *self.baseline_bottom, 1, self.line_width)
        # cv2.line(court, *self.net, 1, self.line_width)
        cv2.line(court, *self.top_inner_line, 1, self.line_width)
        cv2.line(court, *self.bottom_inner_line, 1, self.line_width)
        cv2.line(court, *self.left_court_line, 1, self.line_width)
        cv2.line(court, *self.right_court_line, 1, self.line_width)
        cv2.line(court, *self.left_inner_line, 1, self.line_width)
        cv2.line(court, *self.right_inner_line, 1, self.line_width)
        cv2.line(court, *self.top_second_inner_line, 1, self.line_width)
        cv2.line(court, *self.bottom_second_inner_line, 1, self.line_width)
        cv2.line(court, *self.middle_line_top, 1, self.line_width)
        cv2.line(court, *self.middle_line_bottom, 1, self.line_width)
        court = cv2.dilate(court, np.ones((5, 5), dtype=np.uint8))
        plt.imsave('court_configurations/court_reference.png', court, cmap='gray')
        self.court = court
        return court

    def get_important_lines(self):
        """
        Returns all lines of the court
        """
        lines = [*self.baseline_top, *self.baseline_bottom, *self.net, *self.left_court_line, *self.right_court_line,
                 *self.left_inner_line, *self.right_inner_line, *self.middle_line_top, *self.middle_line_bottom,
                 *self.top_inner_line, *self.bottom_inner_line, *self.top_second_inner_line, *self.bottom_second_inner_line]
        return lines

    def get_extra_parts(self):
        parts = [self.top_extra_part, self.bottom_extra_part]
        return parts

    def save_all_court_configurations(self):
        """
        Create all configurations of 4 points on court reference
        """
        for i, conf in self.court_conf.items():
            c = cv2.cvtColor(255 - self.court, cv2.COLOR_GRAY2BGR)
            for p in conf:
                c = cv2.circle(c, p, 15, (0, 0, 255), 30)
            cv2.imwrite(f'court_configurations/court_conf_{i}.png', c)

    def get_court_mask(self, mask_type=0):
        """
        Get mask of the court
        """
        mask = np.ones_like(self.court)
        if mask_type == 1:  # Bottom half court
            mask[:self.net[0][1] - 500, :] = 0
        elif mask_type == 2:  # Top half court
            mask[self.net[0][1]:, :] = 0
        elif mask_type == 3: # court without margins
            mask[:self.baseline_top[0][1], :] = 0
            mask[self.baseline_bottom[0][1]:, :] = 0
            mask[:, :self.left_court_line[0][0]] = 0
            mask[:, self.right_court_line[0][0]:] = 0
        return mask


if __name__ == '__main__':
    c = CourtReference()
    #c.build_court_reference()
    c.save_all_court_configurations()
