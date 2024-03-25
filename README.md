# Reinforcement_ClashRoyale

## 개요
본 프로젝트는 클래시로얄 게임을 강화학습으로 풀어보고자 하였습니다. 이상적인 학습은 이루어지지 않았지만 강화학습을 위한 ENV 설정과 Reward 체계 구현 등에서 의미를 지니고 있습니다. 동영상으로 구현 동작을 보여주고 싶으나 프로젝트를 한지 너무 오래돼서 괜찮은 동영상이 없고 개발환경 다시 세팅하기에는 시간이 너무 오래 소요될 것 같으므로 문서를 최대한 상세하게 적겠습니다.

## 클래시 로얄이란?
클래시 로얄은 카드 수집형 실시간 타워 디펜스 게임입니다. 사용자는 카드를 선택하여 소환할 수 있으며 정해진 시간안에 상대 킹 타워를 파괴하면 승리합니다. 카드 소환에는 엘릭서라는 자원이 필요하며 카드 덱의 구성에 따라 다양한 전략이 가능합니다.
|game images|cards|
|---|---|
|<img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/games.jpg">|<img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/cards.jpg"> |

## 시작 전
ubuntu 20.04 위에서 개발되었습니다. 클래시 로얄 게임이 ubuntu에서 돌아가지 못하기 때문에 vmware로 windows를 설치하고 그 위에 blue stack을 설치하여 사용합니다.<br>
ubuntu에서 몇가지 에뮬레이터를 설치하여 클래시 로얄을 돌려보려고 하였으나 성공하지 못하였고(개발 초기 단계에서 시간이 오래 지나 정확한 이유 잊어버림) <br>
ubuntu에서 vmware를 설치하고 그 다음 blue stack을 설치하여 게임을 플레이 하였습니다.<br>
arm 기반 architecture가 문제가 되어서 super cell에서 만든 게임을 돌리지 못했던 것으로 기억합니다.<br>
해결 방법이 있으면 알려주세요.


클래시로얄은 CCG(Collectible card Game)이므로 덱의 구성에 따라 전략과 방향성이 달라집니다. 마법카드와 유닛카드의 조합, 앨릭서 생산량, 타워 hp 등에 맞추어 올바르게 설정하세요.

## ENV

설정한 ENV는 다음과 같은 함수를 포함합니다.

 -return_state(img) : 이미지를 가져와 model에 넣을 수 있는 shape로 변경한 후 이미지를 tensor 형태로 반환합니다.<br>
 -check_finish(img) : 현재 게임의 이미지를 가져와 게임이 종료되었는지 확인합니다. 게임이 종료되었으면 1, 종료되지 않았으면 0을 반환합니다.<br>
 -check_win(img) : 현재 게임의 이미지를 가져와 게임이 승리하였는지 확인합니다. 게임 승리 문구가 뜨면 1, 뜨지 않았으면 0을 반환합니다. <br>
 -check_lose(img) : 현재 게임의 이미지를 가져와 게임이 패배하였는지 확인합니다. 게임 패배 문구가 뜨면 1, 뜨지 않았으면 0을 반환합니다. <br>
 -check_card(img) : agent가 카드를 선택하지 않은 상태에서 map을 클릭하였는지 확인합니다. 카드를 선택하지 않은 상태에서 map을 뜨면 '카드를 선택하지 않았습니다' 문구가 뜹니다. 해당 문구를 발견하면 1을 반환합니다.<br>
 -check_elixir(img) : agent가 엘릭서가 부족한 상태에서 카드를 클릭하였는지 확인합니다. 엘릭서가 부족한 상태에서 카드를 클릭하면 '엘릭서가 부족합니다'라는 문구가 뜹니다. 해당 문구를 발견하면 1을 반환합니다.<br>
 -checkET1(img) : 상대 1번 타워의 hp를 확인하여 해당 값을 반환합니다. 타워 hp가 ocr로 인식이 되지 않아 hp바의 길이로 타워 hp를 확인합니다.<br>
 -checkET2(img) : 상대 2번 타워의 hp를 확인하여 해당 값을 반환합니다. 타워 hp가 ocr로 인식이 되지 않아 hp바의 길이로 타워 hp를 확인합니다.<br>
 -checkGameStart(img) : 게임 시작을 판단합니다. 게임 시작 전에 agent가 행동하는 것을 막는 목적으로 활용됩니다.<br>
 -checkMainPage(img) : 메인 페이지에 들어왔음을 확인합니다. 게임을 다시 시작하는 중 잘못 클릭하는 것을 방지하는 역할을 합니다.<br>
 -stop_game(img) : 게임을 종료합니다. 게임을 다시 시작하는 중간 메세지 박스를 확인하는 기능입니다.<br>
 -enemy1(img) : 상대방이 타워를 1개 파괴하였는지 확인합니다. 아군 타워 hp는 글자에 가려져 확인이 어렵기 때문에 파괴됨을 기준으로 reward를 설정합니다.<br>
 -enemy2(img) : 상대방이 타워를 2개 파괴하였는지 확인합니다. 아군 타워 hp는 글자에 가려져 확인이 어렵기 때문에 파괴됨을 기준으로 reward를 설정합니다.<br>
 -enemy3(img) : 상대방이 타워를 3개 파괴하였는지 확인합니다. 아군 타워 hp는 글자에 가려져 확인이 어렵기 때문에 파괴됨을 기준으로 reward를 설정합니다.<br>
 -retryGame : 게임을 다시 시작합니다. 3번의 클릭을 진행하면 게임을 다시 실행할 수 있습니다.<br>

## Agent & Action

Action은 2가지 모드가 존재합니다.<br>
첫째는 카드와 field 선택을 매칭시켜 선택하는 방법입니다. 예를 들면 1번 action은 (1번 카드, field의 1번 구역), 2번 action은 (1번 카드, field의 2번 구역), ... , 7번 action은 (2번 카드, filed의 1번 구역) 이런 식입니다. 전체 액션은 카드 4개 * field 구역 6개 + rest action 1개 해서 25개입니다.<br>
두번째 방법은 카드와 field를 구분하는 방법입니다. 이 방법은 카드 4가지 + field 9가지 + rest action 1개 해서 14가지 action을 가지고 있습니다.<br>
rest action은 아무것도 클릭하지 않는 action입니다.<br>

## Reward
Reward체계는 다음과 같이 구성됩니다.<br>
<p align="center">
 
|screen img|gray_scale|
|---|---|
|단위 시간 당|-1|
|상대 타워 파괴 시|+800|
|카드를 선택하지 않고 field를 선택할 때|-20|
|카드를 클릭하고 filed 선택할 때|+20|
|아군 타워 파괴 시|+400|
|승리|+5000|
|패배|-5000|

</p>

<figure>
 <p align="center">
  <img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/reward.jpg">
 </p>
    
</figure>


## model
팡요랩 PPO code를 참고하여 프로그래밍 하였습니다.<br>
PPO 알고리즘을 사용합니다. 모델 구조는 다음과 같이 구성됩니다. 
<figure>
    <img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/model_img.JPG">
</figure>
행동을 결정하는 actor 모델 구조와 가치를 평가하는 V 모델 구조입니다. actor는 출력단이 action의 개수만큼 적용되어 있으며 categorical과 sample 함수를 통하여 classification과 유사하게 작동합니다. v는 regression과 유사하게 작동합니다.

## state
state는 게임에 영향을 주는 부분을 잘라서 사용합니다. size는 (510, 900, 3)입니다.<br>
사용자 설정에 따라 resize, grayscale, framestack을 적용하여 사용할 수 있습니다.
|screen img|gray_scale|frame_stack|
|---|---|---|
|<img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/play_screen.png">|<img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/play_screen_gray.png"> |<img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/framestack.jpg">|


## graph
|Reward|Loss|
|---|---|
|<img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/Reward_graph.png">|<img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/Loss_graph.png"> |

|click/game|WinLose|
|---|---|
|<img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/click_graph.png">|<img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/winRate_graph.png"> |

## 시연 영상

<figure>
  <p align="center">
    <img src="https://github.com/tuuktuc86/Reinforcement_ClashRoyale/blob/main/cr_test/clash_royale_video.gif" width = "700">
  </p>
</figure>
