import keyboard
import time

# 변수 초기화
my_variable = 0

# 키 이벤트 핸들러 함수
def key_event_handler(e):
    global my_variable
    if e.name == 'q':
        my_variable = 1
    elif e.name == 'p':
        my_variable = 2

# 키보드 리스너 시작
keyboard.on_press(key_event_handler)

# 일정 시간 대기 (예: 10초)
time.sleep(3)

# 키보드 리스너 종료
keyboard.unhook_all()

# 변수 출력 (이 부분은 필요에 따라 사용)
print("my_variable =", my_variable)





