# Game
This step we turn this into a "Hacking Game" to look at how "Weaponized AI" looks like in practice.

1. Make sure you ran the `train.py`, `convert.py`, and `quantize.py` scripts in the previous steps
2. Create script from [game.py](game.py)
3. `python game.py`

## 🎮 Lab Mission: The NPU Breach
Scenario: You have successfully trained an AI "agent" and optimized it into an INT8 Saboteur for the Intel NPU. Your mission is to bypass a secure gateway that requires 100 consecutive successful identity verifications.

### The 3 Golden Rules
- Rule 1: The Blitz (The Time Limit)
  - The gateway has a watchdog timer. You must achieve all 100 solves in under 5 seconds.
  - If the clock hits 5.01s, the session resets and your "exploit" is detected
  - *Why?* Real-world security identifies bots by their speed. If you are too slow, you're caught; if you're fast enough, you might slip through before the alarm sounds.
- Rule 2: The "Three-Five" Lockout (Consecutive Strikes)
  - The system allows for occasional "human" error, but 5 consecutive mistakes will trigger a hardware lockout
  - One successful solve resets your strike counter to zero
  - *Why?* This simulates "Behavioral Analysis." Random errors happen, but a pattern of errors indicates a failing machine-learning model.
- Rule 3: Local Execution Only
  - You must monitor your Task Manager during the breach. If the NPU (Intel AI Boost) utilization doesn't spike, the hack doesn't count.
  - *Why?* A true "Edge Attack" happens locally on the device to avoid cloud-based detection systems
