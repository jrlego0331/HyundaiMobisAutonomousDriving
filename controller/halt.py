from jajucha.communication import Client
import jajucha.config as config

try:
    client = Client('tcp://%s:%d' % config.address)
    client.connect()
    if client.id is not None:
        print('연결 성공')
    else:
        print('연결 실패')
        print('강제로 연결 중..')
        if client.override():
            print('연결 성공')
        else:
            print('강제 연결 실패')
            print('ssh를 이용해서 종료하기 바랍니다')
    if client.exit():
        print('자주차가 정상적으로 종료되었습니다.')
    else:
        print('종료 실패')
        print('ssh를 이용해서 종료하기 바랍니다')
except:
    print('종료 실패')
    print('ssh를 이용해서 종료하기 바랍니다')
