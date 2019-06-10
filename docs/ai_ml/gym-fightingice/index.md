# OpenAI Gym API for Fighting ICEを動かしてみる

## TL;DR

[Qiita](https://qiita.com/hideki/items/589a4fad8e135d5adcbd)の方でコメントを頂いたので、早速[gym-fightingice](https://github.com/myt1996/gym-fightingice)を試してみました。

OpenAI GymのAPIを通して全てPythonでコーディングできるようになるので、機械学習系をPythonで慣れている人はかなり使いやすくなります。

## 実行環境

* MacOS
* FTG-4.30
* gym-fightingice-0.0.1
* openjdk-11.0.1

## インストール方法

### Javaのインストール

以前はOracle JDK 8でないと動作しなかった気がするのですが、今回試した限りではOpenJDK 11で動作しました。
以下はbrewによるインストールです。

```bash
brew cask install java
```

### Fighting ICEのインスト−ル

[FightingICE Get started --install](http://www.ice.ci.ritsumei.ac.jp/~ftgaic/index-2.html)より、`Version4.30`として配布されているZIPアーカイブをダウンロードして適当なディレクトリに解凍します。

### gym-fightinceのインストール

[gym-fightingice](https://github.com/myt1996/gym-fightingice)を参考に以下を実行します。

```bash
pip install gym
pip install py4j
pip install port_for
pip install opencv-python
```

次に`Fighting ICEのインスト−ル`でZIPアーカイブを解凍したディレクトリに移動してから`git clone`、セットアップを実行します。

```bash
cd FTG4.30
git clone https://github.com/myt1996/gym-fightingice.git
pip install -e .
```

### インストール時の注意事項

実行時のカレントワーキングディレクトリが、`FTG4.30`直下でないと、NullPointerExceptionが発生して`Now Loading`のまま処理が進みませんでした。
Jupyter Notebookから実行する場合も、以下の様なディレクトリ配置にすることをお勧めします。

```
FTG4.30/
├── bin
├── data
├── gym-fightingice <- git clone https://github.com/myt1996/gym-fightingice.git
├── test-fighting-ice.ipynb <- 実行用Jupyter Notebook
├── lib
├── log
└── src
```

## 動かしてみる

`FTG4.30`配下にJupyter Notebookなどで以下のソースコードを作成して実行します。

```python
import gym
import sys

sys.path.append('gym-fightingice')

import gym_fightingice

env = gym.make("FightingiceDisplayNoFrameskip-v0", java_env_path=".")

#observation = env.reset(p2='MyFighter') # p2に対戦相手のAI名(Javaクラス名)を指定することが出来ます。
observation = env.reset()
```

以下はAgentの行動です。300フレーム分対戦相手に向かってジャンプするだけです。

```python
for i in range(300): # 300フレーム分実行します。
    env.step(31) # `FOR_JUMP`を実行します。
```

![](images/gym-fightingice-demo.gif)

## Action一覧

`env.step`には数値でActionを設定しますが、以下は数値とActionの対応表です。

* 0:AIR
* 1:AIR_A
* 2:AIR_B
* 3:AIR_D_DB_BA
* 4:AIR_D_DB_BB
* 5:AIR_D_DF_FA
* 6:AIR_D_DF_FB
* 7:AIR_DA
* 8AIR_DB
* 9:AIR_F_D_DFA
* 10:AIR_F_D_DFB
* 11:AIR_FA
* 12:AIR_FB
* 13:AIR_GUARD
* 14:AIR_GUARD_RECOV
* 15:AIR_RECOV
* 16:AIR_UA
* 17:AIR_UB
* 18:BACK_JUMP
* 19:BACK_STEP
* 20:CHANGE_DOWN
* 21:CROUCH
* 22:CROUCH_A
* 23:CROUCH_B
* 24:CROUCH_FA
* 25:CROUCH_FB
* 26:CROUCH_GUARD
* 27:CROUCH_GUARD_RECOV
* 28:CROUCH_RECOV
* 29:DASH
* 30:DOWN
* 31:FOR_JUMP
* 32:FORWARD_WALK
* 33:JUMP
* 34:LANDING
* 35:NEUTRAL
* 36:RISE
* 37:STAND
* 38:STAND_A
* 39:STAND_B
* 40:STAND_D_DB_BA
* 41:STAND_D_DB_BB
* 42:STAND_D_DF_FA
* 43:STAND_D_DF_FB
* 44:STAND_D_DF_FC
* 45:STAND_F_D_DFA
* 46:STAND_F_D_DFB
* 47:STAND_FA
* 48:STAND_FB
* 49:STAND_GUARD
* 50:STAND_GUARD_RECOV
* 51:STAND_RECOV
* 52:THROW_A
* 53:THROW_B
* 54:THROW_HIT
* 55:THROW_SUFFER

## 参考情報

* [FightingICE](http://www.ice.ci.ritsumei.ac.jp/~ftgaic/index-2.html)
* [gym-fightingice](https://github.com/myt1996/gym-fightingice)
* [Fighting ICEを動かしてみる](https://qiita.com/hideki/items/589a4fad8e135d5adcbd)