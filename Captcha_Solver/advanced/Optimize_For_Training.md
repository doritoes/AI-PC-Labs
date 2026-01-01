# Optimize for Training

## Debloat Windows 11
See https://github.com/Raphire/Win11Debloat

Features:
- App Removal
  - remove wide variety of preinstalled apps
  - remove or replace all pinned apps for the current user, or for all existing & new users
- Disable call-home leaks
- Disable Bing web search, Copilot, and AI features
- Personalization to W10-style context menu, streamline settings for lab
- File explorer customized for administrator use
- Taskbar streamlined
- Start menu disable phone link and recommended section
- Disable Xbox and fast startup
- Sysprep mode

Quick Method
- Open PowerShell or Terminal, preferably as an administrator
- Copy and paste the below command into PowerShell
  - `& ([scriptblock]::Create((irm "https://debloat.raphi.re/")))`
- Wait for the script to automatically download Win11Debloat
- Read carefully through and follow on-screen instructions

See the https://github.com/Raphire/Win11Debloat repo:
- how to revert changes
- Traditional method
- Advanced method
- more details

## Purge HP "Ghost" Services
HP Wolf Security and its support services (like HP Sure Sense) are notorious for sticking around even when "disabled." They use kernel-level drivers that keep a footprint in RAM.

To permanently stop the remaining HP services:
- The Nuclear Uninstall Order: If you haven't uninstalled them yet (instead of just disabling), you must do it in this specific order to prevent them from auto-reinstalling via the HP Update Service:
    1. HP Wolf Security
    2. HP Wolf Security - Console
    3. HP Security Update Service
- Force Disable via Services: Press Win + R, type services.msc. Look for the following and set their "Startup Type" to Disabled:
    1. HP Analytics Service (Huge RAM/CPU hog)
    2. HP App Helper Solutions
    3. HP System Info Helper
    4. HP Support Solutions Framework

## Manual RAM Flush & Cache Clearing
Windows 11 aggressively uses "Standby" memory to cache files. For AI training, you want this memory to be "Free," not "Available but Cached."

Option 1 - Microsoft RAMMap (Best Tool): Download [RAMMap](https://learn.microsoft.com/en-us/sysinternals/downloads/rammap) from Microsoft Sysinternals
- Open it, go to the Empty menu, and select Empty Standby List. This instantly flushes the cache and moves "Cached" memory into "Free" memory without a reboot.

Option 2 - Command Line Flush: You can create a desktop shortcut to trigger a background "Idle Task" flush. Set the shortcut path to:
- `%windir%\system32\rundll32.exe advapi32.dll,ProcessIdleTasks`
- Note: This won't show a window; it just forces Windows to perform its cleanup routines immediately.

## Deep Optimization
- Adjust for Best Performance:
    1. Search for "View advanced system settings"
    2. Under Performance > Settings, choose Adjust for best performance
    3. This disables transparency and animations, which significantly reduces the dwm.exe (Desktop Window Manager) RAM footprint
- Disable SysMain (Superfetch): In services.msc, disable SysMain. This service pre-loads apps into RAM based on your usage. For a dedicated AI machine, this is unnecessary and wastes memory.
- Virtual Memory (Pagefile): If your training run hits a "Memory Error," it’s often because Windows is trying to swap to a slow pagefile.
    - Set a Manual Pagefile size (1.5x your total RAM) on your fastest NVMe drive to prevent "stuttering" when RAM fills up.
