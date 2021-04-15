# HGU_Web_Translator

# 고유명사 기호화 및 사전을 이용한 한영 번역 웹 시스템

## 개요

상용 번역기들은 고유명사를 번역할 때 정확하지 않은 경우가 있다.   
본 연구는 Backbone NMT (Neural Machine Translation) 모델에 **고유명사 사전**과 **기호화**를 적용하여 기존의 한/영 번역기에서 “바람과 함께 사라지다” 같은 고유명사를 “Gone With The Wind” 가 아닌 “Vanish With The Wind” 로 번역하는 문제점을 해결하고자 한다.

### 과정
0) 전체 데이터에서 많이 사용된 단어를 추리고 그 중 고유명사를 골라내어 **고유명사 사전**을 만든다. 
1) Backbone NMT 모델 Seq2seq 와 Transformer 에 데이터를 학습 시킬 때, 고유명사 사전에 있는 단어들은 기호로 치환하여 학습한다.
2) 학습된 모델은 고유명사를 직접 번역하지 않고 기호로 치환하여 출력한다.
3) 학습이 완료된 모델에 문장을 넣으면 고유명사 사전에 있는 단어는 기호로 치환되어 모델에 들어가고, 모델이 출력한 문장가 있다면 고유명사 사전에서 해당 단어를 치환하여 최종 문장을 출력한다.

<hr>

## 문제 정의

많은 기업들은 네이버 파파고(papago.naver.com), 구글 번역기 (translate.google.com), 카카오 아이(translate.kakao.com) 와 같이 활발히 딥러닝을 이용한 번역기 연구를 하고 있으며 이는 많은 사람들의 일상 속에 편의를 주고 있다.

<p align="center"> <img width="728" alt="스크린샷 2020-03-27 오후 2 28 34" src="https://user-images.githubusercontent.com/37679062/77724844-4d437480-7037-11ea-8609-07a6943d383e.png"> </p>

상용 번역기들은 우수한 성능을 보이고 있으나, 문장 내에 이름이나 영화 제목과 같은 고유명사를 번역할 때 정확하지 않은 문제가 있다.  
아래의 번역은 2019년 11월 25일 기준 결과이다.

예시)
입력 문장: 어제 조카랑 ‘겨울 왕국’을 봤어.
  - 파파고 번역: I saw ‘Winter Kingdom’ with my nephew yesterday.
  - 구글 번역기: I saw your niece and the Winter Kingdom yesterday.
  
상용 번역기에서 ‘겨울 왕국(Frozen)’ 을 'Winter Kingdom' 으로 오번역 하고 있다.

<hr>

## Conceptual Design

<p align="center"> <img width="585" alt="스크린샷 2020-03-27 오후 2 31 37" src="https://user-images.githubusercontent.com/37679062/77725017-baefa080-7037-11ea-9151-07481bfb01de.png"> </p>

__사용된 기술:__

<p align="center"> <img width="585" alt="스크린샷 2020-03-27 오후 2 31 37" src="https://user-images.githubusercontent.com/37679062/77750627-77ad2600-7067-11ea-8ad0-0e225778482a.JPG"> </p>

*학습 모델에 `Transformer` 도 포함.

<hr>

## 고유명사 사전 기호화

숫자 기호화와 비슷한 구조를 가진다. 기본적인 과정은 다음과 같다. 번역 모델이 번역할 때 입력으로 하나의 문장이 들어오면, 문장에 있는 고유명사를 기호화 한다. 번역 후 남아있는 기호를 입력문장에 있는 고유명사를 이용해 고유명사 딕셔너리 사전을 확인한 후 다시 원래 고유명사로 바꾼다.

__1. 모델을 학습할 때__

(1)인용 부호 ‘ ’로 고유명사 치환
인용 부호안에 들어가 있는 단어를 고유명사로 치환한다. 예를 들어 ‘겨울 왕국'과 같은 경우 일반적인 ‘Winter Kingdom’이 아닌 고유 명사 취급하여 ‘frozen’으로 번역되게끔 한다.

(2)대문자로 된 명사를 고유명사 치환
대문자로 시작하는 명사를 고유명사로 치환한다. 예를 들어 “Kim is a father” 라는 문장이 입력으로 들어오면 대문자로 된 명사를 기호화하여 입력을 “_N2 is a father”로 바꾸고, 그 후 번역 모델로 번역하여 “_N2 는 아빠이다"라는 번역 결과를 얻고, 기호를 고유명사 사전을 통해 바꾸어 “김씨는 아빠이다"라는 최종 번역문을 얻는다.

(3)입력문장 고유명사 classifier
입력문장에서 고유명사를 구별하기 위한 classifier 를 만들어 해당 문장을 번역시 고유명사 classifier 로 고유명사 부분을 찾아 따로 고유명사 dictionary 를 사용해 번역을 한다.

(4)입력문장에서 고유명사가 없을 경우
입력문장에서 고유명사가 없을 경우 모든 명사를 고유명사로 취급한 후 고유명사 사전에 포함되어 있으면 고유명사로 치환, 고유명사 사전에 포함되어 있지 않는 경우 일반 명사로 취급하여 번역한다. 이를 통해 인용 부호나 대문자로 되어있지 않은 고유명사를 찾을 수 있다.

__2. 번역할 때__

학습이 끝난 후 새로운 문장을 번역(inference) 할 때는 학습 데이터를 다룰 때와는 다른 알고리즘을 적용한다. 학습할 때와 달리 정답이 없는 입력 문장만 주어지기 때문에 입력 문장의 고유명사를 기호화하고 모델에 넣어 번역한다. 나온 번역문의 기호를 다시 원래 고유명사로 치환한다.

<hr>

## 실험 순서

1. Tokenize 진행 (./tokenize.perl)
2. 고유명사 처리 
3. BPE 및 subwording 작업 (./data_prepare.sh)
4. 학습하기 (./train.sh)
5. 학습후 data cleansing (./two_gram_bleu) 
6. 다시 학습
7. 웹 사용(트랜슬레이션) - 준비물 : 학습된 모델

<hr>


## 결과

<p align="center"> <img width="690" alt="스크린샷 2020-03-27 오후 2 41 27" src="https://user-images.githubusercontent.com/37679062/77725542-0fdfe680-7039-11ea-9fc3-9cae513bee3e.png"> </p>

해당 번역기는 http://www.aihub.or.kr/ 에서 한국어-영어 번역 말뭉치 데이터(약 160만 문장)를 받아와 해당 데이터에서 제일 흔히 나오는 450의 고유명사를 기호화 하여 학습한 모델로 진행했습니다.

고유명사 기호화를 사용하지 않았을때의 Bleu score: 26.55

고유명사 기호화를 사용했을 때의 Bleu score: 28.48





