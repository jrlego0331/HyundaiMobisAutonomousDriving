from jajucha.planning import BasePlanning
from jajucha.graphics import Graphics
import cv2
import numpy as np
import time


class Planning(BasePlanning):
    def __init__(self, graphics):
        super().__init__(graphics)
        # --------------------------- #
        self.vars.counter = 0  # 변수 설정

    def process(self, t, frontImage, rearImage, frontLidar, rearLidar):
        """
        자주차의 센서 정보를 바탕으로 조향과 속도를 결정하는 함수
        t: 주행 시점으로부터의 시간 (초)
		frontImage: 전면 카메라 이미지
		rearImage: 후면 카메라 이미지
		frontLidar: 전면 거리 센서 (mm), 0은 오류를 의미함
		rearLidar: 후면 거리 센서 (mm), 0은 오류를 의미함
        """
        frontLines, frontObject = self.processFront(frontImage) # 전면 카메라 이미지 처리
        rearLines = self.processRear(rearImage) # 후면 카메라 이미지 처리

        self.vars.counter += 1 # 변수 수정
        print(self.vars.counter) # 변수 출력

        steer = -15
        velocity = 40
        return steer, velocity


if __name__ == "__main__":
    g = Graphics(Planning) # 자주차 컨트롤러 실행
    g.root.mainloop() # 클릭 이벤트 처리
    g.exit() # 자주차 컨트롤러 종료
