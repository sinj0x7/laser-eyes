# 👁️ laser-eyes

impress ur discord friends with real-time laser eyes on your webcam.

uses mediapipe to track your irises and shoots glowing beams from your eyes through a virtual camera that discord sees as a real webcam.



## how it works

```
real webcam → mediapipe face tracking → laser effect overlay → ffmpeg → virtual cam → discord
```

press **SPACE** to charge up, lasers fire automatically at 100%.

## setup

**1. install deps**
```bash
pip install opencv-python mediapipe
sudo dnf install -y ffmpeg akmod-v4l2loopback   # fedora
# sudo apt install ffmpeg v4l2loopback-dkms      # ubuntu/debian
```

**2. load virtual camera**
```bash
sudo modprobe v4l2loopback video_nr=4 card_label="LaserEyes" exclusive_caps=1
```

**3. download face model**
```bash
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**4. run**
```bash
python main.py
```

**5. in discord**

Settings → Voice & Video → Camera → pick **LaserEyes**

## controls

| key | action |
|-----|--------|
| `SPACE` | charge & fire |
| `Q` | quit |

## effects

- 👁️ iris glow that builds as you charge
- ⚡ rotating charge ring around each eye  
- 🔴 neon purple/white laser beams shooting from eyes
- 📳 screen shake on fire
- 💥 edge pulse flash

## requirements

- linux (uses v4l2loopback for virtual cam)
- python 3.10+
- webcam