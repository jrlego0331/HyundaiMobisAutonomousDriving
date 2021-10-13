from jajucha.planning import BasePlanning
from jajucha.graphics import Graphics
from jajucha.control import mtx
import cv2
import numpy as np
import time

class Planning(BasePlanning):
    def __init__(self, graphics):
        super().__init__(graphics)
        # --------------------------- # 초기 변수를 설정하는 함수
        self.vars.redCnt = 0  # 빨강불 카운트 변수 설정
        self.vars.greenCnt = 0  # 녹색불 카운트 변수 설정
        self.vars.stop = True # 차가 처음에 정지해 있음을 정의한 변수 설정
        self.vars.steer = -8  # 조향값은 0인 상태로 변수 설정
        self.vars.velocity = 0  # 속도도 0인 상태로 변수 설정
        
        #PID control var
        self.error_current = 0
        self.HW_error_calibration = -8
        self.error_min_threshold = 20
        self.error_max_threshold = 100

    def PID(self, kP = 35/100, kD = 0/100):

        P = self.error_current * kP

        steer = P + self.HW_error_calibration

        if abs(self.vars.steer - steer) < self.error_min_threshold:
            return self.vars.steer + steer * 0.4
        if abs(self.vars.steer - steer) > self.error_max_threshold:
            return self.vars.steer 
        if steer > 55:
            steer = 55
        if steer < -55:
            steer = -55
        return steer

    def process(self, t, frontImage, rearImage, frontLidar, rearLidar):
        """
        자주차의 센서 정보를 바탕으로 조향과 속도를 결정하는 함수
        t: 주행 시점으로부터의 시간 (초)
        frontImage: 전면 카메라 캘리된 이미지(640X480)
        rearImage: 후면 카메라 캘리된 이미지
        frontLidar: 전면 거리 센서 (mm), 0은 오류를 의미함, 0<x<2000 (2m 최대거리)
        rearLidar: 후면 거리 센서 (mm), 0은 오류를 의미함
        """

        laneParameterY = 360   # 레인 확인 파라미터
        laneParameterX = 260
        linecheck = 380 # no450

        frontLines, frontObject = self.processFront(frontImage)  # 전면 카메라 이미지 처리
        # frontLines = [[x1,y1],[x2,y2],[x3,y3], … ] , [[x1,y1], [x2,y2], … ]
        # frontLines[0]이 가장 왼쪽 차선, 빨(0), 주(1), 노(2), 초(3), 파(4), 남(5), 보(6)
        #rearLines= self.processRear(rearImage) # 후면 카메라 이미지 처리

        # 신호등 처리
        reds, greens = frontObject # reds : n*3의 크기
        # reds: numpy array([[x1,y1,반지름], [x2,y2,반지름], ...])

        # canny image 출력 
        canny = self.canny(frontImage)

        # 분석을 위한 y값 설정 
        x = 320
        y = 479
        while y >= 0:
            if canny[y, x] > 0:
                break
            y -= 1
        # print('479-y=', 479-y)
        
        # 차선 선택
        center_x = mtx[0, 2]       # 이미지의 가운데, center_x=320 부근의 값을 가지게 됩니다.
        # print ('center_x=', center_x)
        line = None
        frontLines.sort(key=lambda x:x[0, 1], reverse=True)
        # frontLines를 정렬(x[0,1]을 기준으로 내림차순으로 정렬)
        # frontLines = [[x1,y1],[x2,y2],[x3,y3], … ] , [[x1,y1], [x2,y2], … ]
        # y1>y2>y3... : 밑의 점부터 표현
        # print ('frontLines=', frontLines)

        for i in range(len(frontLines)):
            if frontLines[i][0, 1] < laneParameterY: # (1) [No 460]은 적절한 값이 아닙니다!! / y1, y2, y3... [No 460]보다 작은가? 
                continue              # y값이 [No 460]보다 작으면 추세선 안하고 넘어가라(479~[No 460]까지만 추세선 찾기)
            x = frontLines[i][:, 0]    # x=[x1, x2, x3, ...]
            y = frontLines[i][:, 1]    # y=[y1, y2, y3, ...]
            coefficient = np.polyfit(y, x, 1)    # coefficient = [a, b] (단, x = ay + b) 
            line = np.poly1d(coefficient)        # line = ay + b (가장 밑에 닿은 차선의 추세선 식)

            if line(linecheck) < center_x:   # (2) [No 450]은 적절한 값이 아닙니다!! / 
                      # 만약 line = a*[No 450] + b : y=[No 450]에서의 x값이 중앙보다 왼쪽이면
                line = 'left', i            # line = ('left', i) 인 class, tuple
                break
            else:                           # 만약 line = a*400 + b : y=400에서의 x값이 중앙보다 왼쪽이 아니면
                line = 'right', i           # line = ('right', i) 인 class, tuple
                break
        # print('line', line)
        #laneImage = frontImage.copy()
        
        #보더라인 출력
        '''
        cv2.line(laneImage, (320+ laneParameterX, 480), (320+ laneParameterX, laneParameterY), (20, 255, 20), 2)
        cv2.line(laneImage, (320- laneParameterX, 480), (320- laneParameterX, laneParameterY), (20, 255, 20), 2)
        cv2.line(laneImage, (0, laneParameterY), (640, laneParameterY), (255, 20, 20), 2)
        cv2.line(laneImage, (0, linecheck), (640, linecheck), (20, 255, 20), 2)
        cv2.line(laneImage, (320, 0), (320, 480), (50, 255, 50), 2)
'''
        if line is None:                    # line 이 없으면
            self.error_current = 0
            if self.vars.steer >= 0:
                self.vars.steer = 80
                self.vars.velocity = 45
                return self.vars.steer, self.vars.velocity
            elif self.vars.steer <= 0:
                self.vars.velocity = 45
                self.vars.steer = -80
                return self.vars.steer, self.vars.velocity
            
        elif line[0] == 'right':              # line 이 우차선이면
            # follow right
            # left = frontLines[line[1]-1]
            line = frontLines[line[1]]    # 다시 line 재정의 
            x = line[:, 0]
            y = line[:, 1]
            coefficient = np.polyfit(y, x, 1)
            line = np.poly1d(coefficient)

            e = line(linecheck) - center_x - laneParameterX
            self.error_current = e
            
          #  for i in range(200, 480, 20):
                #cv2.circle(laneImage, (int(line(i)), i), 5, (255, 0, 0), -1)
            # cv2.circle(frontImage, (int(f(400)), 400), 5, (255, 0, 0), -1)
            #cv2.imshow('Front Lane Image', laneImage)
        else:
            # follow left
            line = frontLines[line[1]]
            x = line[:, 0]
            y = line[:, 1]
            coefficient = np.polyfit(y, x, 1)
            line = np.poly1d(coefficient)
            
            e = line(linecheck) - center_x + laneParameterX
            self.error_current = e

            #for i in range(200, 480, 20):
                #cv2.circle(laneImage, (int(line(i)), i), 5, (255, 0, 0), -1)
            # cv2.circle(frontImage, (int(f(400)), 400), 5, (255, 0, 0), -1)
            #cv2.imshow('Front Lane Image', laneImage)

        #print('line(', linecheck, ')', line(linecheck)) # (2) [No 450]의 적절한 값을 넣어주세요.
        #print('center_x=', center_x, '\ne=', e)

        steer =  self.PID() # (5) [No 2]와 (6) [No 3]은 적절한 값이 아닙니다. steer는 조향값이며, e는 픽셀의 차이입니다.
        # 픽셀과 조향값과는 어떤 관계가 있을까요? 이 관계를 가장 적절한 1차식으로 만들어 봅시다.
        # 참고로 조향값 steer는 -80~80 을 추천합니다. [-100~100] 사이값은 이론적인 값이며 실제 이 값을 넣으면
        # 기계에 무리가 옵니다. 절대 넣지 않기를 조언합니다.

        velocity = 60     # [No 10]은 적절한 값이 아닐 수 있습니다. velocity는 속도입니다.
        # -50 ~ 50의 값을 추천합니다. [-300 ~ 300] 사이값은 이론적인 값이며
        # 실제 이 값을 넣으면 기계에 무리가 옵니다. 절대 넣지 않기를 조언합니다.
        # 이 velocity를 e값의 1차 함수로 넣을 수도 있습니다.
        canny = None
        frontImage = None

        # cv2.waitKey(1)
        if steer > -80 and steer < 80:
            self.vars.steer = steer
        self.vars.velocity = velocity
        return self.vars.steer, self.vars.velocity

if __name__ == "__main__":
    g = Graphics(Planning) # 자주차 컨트롤러 실행
    g.root.mainloop() # 클릭 이벤트 처리
    g.exit() # 자주차 컨트롤러 종료
