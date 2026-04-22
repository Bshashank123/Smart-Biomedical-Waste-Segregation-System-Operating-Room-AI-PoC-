#!/usr/bin/env python3
"""
Smart Biomedical Waste Segregation — Surgery Room PoC
======================================================
Designed specifically for Operating Room (OR) waste classification.
Uses YOLOv8l (large model) + image preprocessing + multi-pass
detection for maximum accuracy.

Bins:
  🔴 Red    — Infectious / Contaminated Waste
  🟡 Yellow — Sharps Waste
  🔵 Blue   — Plastic / Glass / Pharmaceutical Waste
  🟢 Green  — General / Non-Infectious Waste

Usage:
    pip install ultralytics opencv-python requests

    # Built-in webcam (index 0)
    python smart_waste_poc.py

    # DroidCam / IP Webcam  ─ pick ONE of the URL forms:
    python smart_waste_poc.py --droidcam 192.168.1.42
    python smart_waste_poc.py --droidcam 192.168.1.42:4747          # custom port
    python smart_waste_poc.py --droidcam http://192.168.1.42:4747   # full URL
    python smart_waste_poc.py --droidcam rtsp://192.168.1.42:4747/h264_pcm.sdp

    # Still image
    python smart_waste_poc.py --image x.jpg

    # Adjust sensitivity
    python smart_waste_poc.py --conf 0.30
"""

import argparse
import sys
import time
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL      = "yolov8l.pt"    # large model — best accuracy (auto-downloaded ~87 MB)
CONF       = 0.30            # lower threshold catches more surgical items
IOU        = 0.45            # NMS IoU — tighter = fewer duplicate boxes
IMG_SIZE   = 832             # larger input = finer detail detection (default 640)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── DroidCam / IP Webcam defaults ──────────────────────────────────────────
DROIDCAM_PORT      = 4747    # DroidCam default HTTP/RTSP port
DROIDCAM_HTTP_PATH = "/video"          # DroidCam MJPEG endpoint
IPWEBCAM_HTTP_PATH = "/videofeed"      # IP Webcam (Android) MJPEG endpoint
# Connection tuning for wireless cameras
DROIDCAM_OPEN_TIMEOUT  = 8     # seconds to wait for first frame
DROIDCAM_READ_TIMEOUT  = 5000  # ms between frames before giving up
# ──────────────────────────────────────────────────────────────────────────

STEPS = [
    "Show waste to camera",
    "Image captured",
    "AI classifies waste",
    "System selects category",
    "LED indicates bin",
    "Waste disposed",
]

# ─────────────────────────────────────────────
#  BIN DEFINITIONS
# ─────────────────────────────────────────────
BINS = {
    "Red": {
        "emoji": "🔴", "label": "Infectious Waste",
        "desc":  "Contaminated / body-fluid-exposed / biological items",
        "bgr":   (30, 30, 220), "ansi": "\033[91m",
    },
    "Yellow": {
        "emoji": "🟡", "label": "Sharps Waste",
        "desc":  "Needles, blades, trocars — anything that punctures or cuts",
        "bgr":   (0, 200, 230), "ansi": "\033[93m",
    },
    "Blue": {
        "emoji": "🔵", "label": "Plastic / Glass / Pharma Waste",
        "desc":  "Vials, bottles, IV bags, ampoules, blister packs",
        "bgr":   (200, 80, 0),  "ansi": "\033[94m",
    },
    "Green": {
        "emoji": "🟢", "label": "General / Non-Infectious Waste",
        "desc":  "Packaging, paper, non-contaminated wrappers",
        "bgr":   (0, 160, 0),   "ansi": "\033[92m",
    },
}

# ─────────────────────────────────────────────
#  SURGERY ROOM WASTE MAP
#  Covers every item commonly found in an OR
# ─────────────────────────────────────────────
WASTE_MAP: dict[str, str] = {

    # ════════════════════════════════════════
    # 🟡 YELLOW — SHARPS WASTE
    # ════════════════════════════════════════
    "scalpel": "Yellow", "scalpel blade": "Yellow", "surgical blade": "Yellow",
    "knife": "Yellow", "scissors": "Yellow", "surgical scissors": "Yellow",
    "mayo scissors": "Yellow", "metzenbaum scissors": "Yellow",
    "iris scissors": "Yellow", "stitch scissors": "Yellow",
    "suture scissors": "Yellow", "dissecting scissors": "Yellow",
    "bandage scissors": "Yellow", "needle": "Yellow",
    "suture needle": "Yellow", "curved needle": "Yellow",
    "straight needle": "Yellow", "cutting needle": "Yellow",
    "tapered needle": "Yellow", "hypodermic needle": "Yellow",
    "injection needle": "Yellow", "spinal needle": "Yellow",
    "epidural needle": "Yellow", "biopsy needle": "Yellow",
    "trocar": "Yellow", "trocar tip": "Yellow",
    "laparoscopic trocar": "Yellow", "cannula": "Yellow",
    "sharp cannula": "Yellow", "veress needle": "Yellow",
    "insufflation needle": "Yellow", "lancet": "Yellow",
    "surgical staple": "Yellow", "staple": "Yellow",
    "skin stapler": "Yellow", "bone saw": "Yellow",
    "oscillating saw": "Yellow", "sagittal saw": "Yellow",
    "wire": "Yellow", "kirschner wire": "Yellow", "k-wire": "Yellow",
    "bone drill bit": "Yellow", "drill bit": "Yellow",
    "awl": "Yellow", "curette": "Yellow", "osteotome": "Yellow",
    "chisel": "Yellow", "fork": "Yellow",
    "toothpick": "Yellow", "toothbrush": "Yellow",

    # ════════════════════════════════════════
    # 🔴 RED — INFECTIOUS / CONTAMINATED WASTE
    # ════════════════════════════════════════
    "glove": "Red", "surgical glove": "Red", "latex glove": "Red",
    "nitrile glove": "Red", "examination glove": "Red",
    "sterile glove": "Red", "mask": "Red", "surgical mask": "Red",
    "n95 mask": "Red", "face mask": "Red", "respirator": "Red",
    "face shield": "Red", "visor": "Red", "gown": "Red",
    "surgical gown": "Red", "isolation gown": "Red", "apron": "Red",
    "surgical apron": "Red", "drape": "Red", "surgical drape": "Red",
    "sterile drape": "Red", "fenestrated drape": "Red", "cap": "Red",
    "surgical cap": "Red", "bouffant cap": "Red", "hair cover": "Red",
    "hood": "Red", "shoe cover": "Red", "boot cover": "Red",
    "overshoe": "Red", "protective eyewear": "Red", "goggles": "Red",
    "gauze": "Red", "gauze pad": "Red", "gauze sponge": "Red",
    "surgical sponge": "Red", "lap sponge": "Red",
    "laparotomy sponge": "Red", "swab": "Red", "cotton swab": "Red",
    "cotton ball": "Red", "cotton roll": "Red", "bandage": "Red",
    "dressing": "Red", "wound dressing": "Red",
    "adhesive dressing": "Red", "non-adherent dressing": "Red",
    "abdominal pad": "Red", "combine dressing": "Red",
    "telfa pad": "Red", "hemostatic gauze": "Red", "surgicel": "Red",
    "gelfoam": "Red", "bone wax": "Red", "catheter": "Red",
    "urinary catheter": "Red", "foley catheter": "Red",
    "central line": "Red", "central venous catheter": "Red",
    "arterial line": "Red", "chest tube": "Red", "drain": "Red",
    "surgical drain": "Red", "jackson-pratt drain": "Red",
    "hemovac drain": "Red", "nasogastric tube": "Red",
    "endotracheal tube": "Red", "et tube": "Red",
    "tracheostomy tube": "Red", "suction catheter": "Red",
    "suction tube": "Red", "yankauer": "Red", "suction tip": "Red",
    "suction canister": "Red", "drainage bag": "Red",
    "urine bag": "Red", "colostomy bag": "Red", "stoma bag": "Red",
    "blood bag": "Red", "specimen bag": "Red",
    "specimen container": "Red", "sample container": "Red",
    "biopsy specimen": "Red", "tissue sample": "Red",
    "pathology specimen": "Red", "blood": "Red",
    "blood-soaked": "Red", "fluid-soaked": "Red",
    "contaminated": "Red", "soiled": "Red", "suture": "Red",
    "suture thread": "Red", "absorbable suture": "Red",
    "non-absorbable suture": "Red", "vicryl": "Red", "prolene": "Red",
    "nylon suture": "Red", "silk suture": "Red",
    "chromic suture": "Red", "monofilament": "Red",
    "surgical thread": "Red", "ligature": "Red",
    "wound closure strip": "Red", "steri-strip": "Red",
    "electrocautery tip": "Red", "cautery tip": "Red",
    "bovie tip": "Red", "electrosurgical pencil": "Red",
    "diathermy pad": "Red", "grounding pad": "Red",
    "electrosurgery pad": "Red", "forceps": "Red",
    "tissue forceps": "Red", "thumb forceps": "Red",
    "hemostat": "Red", "hemostatic forceps": "Red",
    "kelly clamp": "Red", "mosquito clamp": "Red",
    "allis clamp": "Red", "babcock clamp": "Red",
    "towel clip": "Red", "towel clamp": "Red",
    "needle holder": "Red", "mayo-hegar": "Red", "retractor": "Red",
    "army navy retractor": "Red", "richardson retractor": "Red",
    "deaver retractor": "Red", "self-retaining retractor": "Red",
    "balfour retractor": "Red", "weitlaner retractor": "Red",
    "speculum": "Red", "vaginal speculum": "Red",
    "nasal speculum": "Red", "uterine curette": "Red",
    "bone curette": "Red", "rongeur": "Red", "bone rongeur": "Red",
    "elevator": "Red", "periosteal elevator": "Red", "mallet": "Red",
    "surgical mallet": "Red", "probe": "Red", "surgical probe": "Red",
    "used packaging": "Red", "contaminated wrap": "Red",
    "bird": "Red", "cat": "Red", "dog": "Red", "horse": "Red",
    "cow": "Red", "bear": "Red", "banana": "Red", "apple": "Red",
    "orange": "Red", "broccoli": "Red", "carrot": "Red",
    "hot dog": "Red", "pizza": "Red", "donut": "Red", "cake": "Red",
    "sandwich": "Red", "bowl": "Red", "spoon": "Red",

    # ════════════════════════════════════════
    # 🔵 BLUE — PLASTIC / GLASS / PHARMA WASTE
    # ════════════════════════════════════════
    "iv bag": "Blue", "iv fluid bag": "Blue", "infusion bag": "Blue",
    "saline bag": "Blue", "drip bag": "Blue", "iv bottle": "Blue",
    "infusion bottle": "Blue", "vial": "Blue", "glass vial": "Blue",
    "plastic vial": "Blue", "medicine vial": "Blue",
    "drug vial": "Blue", "ampoule": "Blue", "glass ampoule": "Blue",
    "medicine bottle": "Blue", "drug bottle": "Blue", "bottle": "Blue",
    "glass bottle": "Blue", "plastic bottle": "Blue",
    "reagent bottle": "Blue", "saline bottle": "Blue",
    "antiseptic bottle": "Blue", "betadine bottle": "Blue",
    "alcohol bottle": "Blue", "iodine bottle": "Blue",
    "hydrogen peroxide bottle": "Blue", "eye drop bottle": "Blue",
    "nasal spray bottle": "Blue", "syringe": "Blue",
    "syringe barrel": "Blue", "syringe plunger": "Blue",
    "prefilled syringe": "Blue", "empty syringe": "Blue",
    "insulin syringe": "Blue", "blister pack": "Blue",
    "medicine pack": "Blue", "pill pack": "Blue", "foil pack": "Blue",
    "tablet pack": "Blue", "capsule pack": "Blue",
    "drug packaging": "Blue", "sterile packaging": "Blue",
    "peel pack": "Blue", "tyvek pouch": "Blue", "medicine cup": "Blue",
    "gallipot": "Blue", "kidney dish": "Blue", "basin": "Blue",
    "specimen cup": "Blue", "urine cup": "Blue",
    "graduated cup": "Blue", "cup": "Blue", "wine glass": "Blue",
    "glass": "Blue", "vase": "Blue", "beaker": "Blue", "flask": "Blue",
    "test tube": "Blue", "petri dish": "Blue",
    "blood collection tube": "Blue", "vacutainer": "Blue",
    "sample tube": "Blue", "luer lock": "Blue",
    "extension set": "Blue", "stopcock": "Blue",
    "oxygen mask": "Blue", "nebuliser mask": "Blue",
    "oxygen tubing": "Blue", "breathing circuit": "Blue",

    # ════════════════════════════════════════
    # 🟢 GREEN — GENERAL / NON-INFECTIOUS WASTE
    # ════════════════════════════════════════
    "paper": "Green", "paper towel": "Green", "tissue": "Green",
    "tissue paper": "Green", "newspaper": "Green",
    "cardboard": "Green", "cardboard box": "Green", "book": "Green",
    "notepad": "Green", "label": "Green", "sticker": "Green",
    "instruction leaflet": "Green", "patient notes": "Green",
    "outer packaging": "Green", "wrapper": "Green",
    "plastic wrapper": "Green", "shrink wrap": "Green",
    "cling film": "Green", "packaging foam": "Green",
    "bubble wrap": "Green", "plastic bag": "Green",
    "zip lock bag": "Green", "sealed bag": "Green",
    "sterile pouch outer": "Green", "laptop": "Green",
    "keyboard": "Green", "mouse": "Green", "tv": "Green",
    "monitor": "Green", "remote": "Green", "cell phone": "Green",
    "tablet": "Green", "calculator": "Green", "clock": "Green",
    "battery": "Green", "chair": "Green", "couch": "Green",
    "bed": "Green", "table": "Green", "dining table": "Green",
    "sink": "Green", "toilet": "Green", "potted plant": "Green",
    "microwave": "Green", "refrigerator": "Green", "oven": "Green",
    "toaster": "Green", "hair drier": "Green", "backpack": "Green",
    "handbag": "Green", "suitcase": "Green", "umbrella": "Green",
    "tie": "Green", "teddy bear": "Green", "sports ball": "Green",
    "frisbee": "Green", "kite": "Green", "baseball bat": "Green",
    "skateboard": "Green", "surfboard": "Green",
    "tennis racket": "Green", "skis": "Green", "snowboard": "Green",
    "bicycle": "Green", "car": "Green", "motorcycle": "Green",
    "airplane": "Green", "bus": "Green", "train": "Green",
    "truck": "Green", "boat": "Green",

    # Non-waste / skip
    "person": None, "bench": None, "traffic light": None,
    "fire hydrant": None, "stop sign": None, "parking meter": None,
}

# ─────────────────────────────────────────────
#  KEYWORD FALLBACK
# ─────────────────────────────────────────────
_KEYWORD_RULES: list[tuple[list[str], str]] = [
    (["needle","syringe with needle","blade","lancet","scalpel","trocar",
      "wire","staple","bone saw","drill","awl","chisel","sharp"], "Yellow"),
    (["glove","mask","gown","drape","gauze","sponge","swab","bandage",
      "dressing","suture","catheter","drain","tube","soiled","contaminated",
      "blood","specimen","tissue","biopsy","cautery","forceps","clamp",
      "retractor","hemostat"], "Red"),
    (["vial","ampoule","bottle","bag","iv","infusion","blister","pack",
      "syringe","cup","beaker","flask","test tube","petri","vacutainer",
      "oxygen","nebuli","breathing"], "Blue"),
    (["paper","cardboard","box","wrapper","packaging","foam","book",
      "label","sticker"], "Green"),
]

def classify(label: str) -> str:
    label = label.lower().strip()
    if label in WASTE_MAP:
        result = WASTE_MAP[label]
        return result if result else "Green"
    for keywords, bin_color in _KEYWORD_RULES:
        if any(kw in label for kw in keywords):
            return bin_color
    return "Green"


# ─────────────────────────────────────────────
#  DROIDCAM / IP WEBCAM  URL BUILDER
# ─────────────────────────────────────────────
def resolve_camera_source(arg: str) -> str | int:
    """
    Accepts any of:
      • bare integer          → local webcam index  (e.g. "0")
      • bare IP               → DroidCam HTTP MJPEG (e.g. "192.168.1.42")
      • IP:port               → DroidCam HTTP MJPEG (e.g. "192.168.1.42:4747")
      • http://...            → kept as-is (MJPEG or HLS)
      • rtsp://...            → kept as-is (H.264 stream)

    DroidCam default   : http://<ip>:4747/video
    IP Webcam (Android): http://<ip>:8080/videofeed
    EpocCam            : http://<ip>:2431/video  or RTSP
    """
    arg = arg.strip()

    # Already a full URL?
    if re.match(r"^(rtsp|rtsps|http|https)://", arg, re.IGNORECASE):
        print_info(f"Using stream URL: {arg}")
        return arg

    # Bare integer → local device index
    if arg.isdigit():
        return int(arg)

    # IP[:port] → build DroidCam HTTP URL
    host, _, port = arg.partition(":")
    port = port or str(DROIDCAM_PORT)
    url  = f"http://{host}:{port}{DROIDCAM_HTTP_PATH}"
    print_info(f"DroidCam detected — connecting to: {url}")
    return url


def open_camera(source: str | int) -> cv2.VideoCapture:
    """Open a VideoCapture with sensible settings for IP streams."""
    if isinstance(source, str) and source.startswith(("rtsp://", "rtsps://")):
        # For RTSP prefer TCP to avoid UDP packet loss over Wi-Fi
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, DROIDCAM_OPEN_TIMEOUT * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, DROIDCAM_READ_TIMEOUT)
    elif isinstance(source, str) and source.startswith("http"):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, DROIDCAM_OPEN_TIMEOUT * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, DROIDCAM_READ_TIMEOUT)
        # Keep buffer small — we want the latest frame, not a queue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def verify_camera(cap: cv2.VideoCapture, source) -> None:
    """Check the capture opened and can deliver a frame; exit with hint if not."""
    if not cap.isOpened():
        msg = f"❌  Cannot open camera: {source}"
        if isinstance(source, str) and "192.168" in source:
            msg += (
                "\n\n  Troubleshooting DroidCam / IP Webcam:"
                "\n  • Phone and PC must be on the same Wi-Fi network"
                "\n  • Open DroidCam on your phone first, then run this script"
                "\n  • Try the USB connection option in DroidCam for lower latency"
                "\n  • Try IP Webcam app path:  --droidcam http://<ip>:8080/videofeed"
                "\n  • Check firewall is not blocking the port"
            )
        sys.exit(msg)

    ret, _ = cap.read()
    if not ret:
        cap.release()
        sys.exit(
            f"❌  Opened {source} but received no frames.\n"
            "    Make sure the DroidCam / IP Webcam app is running and streaming."
        )


# ─────────────────────────────────────────────
#  IMAGE PRE-PROCESSING
# ─────────────────────────────────────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
    return denoised


# ─────────────────────────────────────────────
#  MULTI-PASS DETECTION
# ─────────────────────────────────────────────
def multi_pass_detect(model: YOLO, frame: np.ndarray,
                       conf: float, iou: float) -> list[dict]:
    processed = preprocess(frame)
    all_boxes  = []

    for imgsz in [IMG_SIZE, 640]:
        results = model(
            processed,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            augment=True,
        )[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id].lower().replace("-", " ")
            conf_  = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            bin_   = classify(label)
            if bin_ is None:
                continue
            all_boxes.append({
                "label": label, "conf": conf_,
                "bin": bin_, "bbox": (x1, y1, x2, y2),
            })

    return _nms_deduplicate(all_boxes, iou_thresh=0.50)


def _iou(a: tuple, b: tuple) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def _nms_deduplicate(boxes: list[dict], iou_thresh: float) -> list[dict]:
    boxes = sorted(boxes, key=lambda d: d["conf"], reverse=True)
    kept  = []
    for b in boxes:
        if all(_iou(b["bbox"], k["bbox"]) < iou_thresh for k in kept):
            kept.append(b)
    return kept


# ─────────────────────────────────────────────
#  TERMINAL HELPERS
# ─────────────────────────────────────────────
R = "\033[0m"; B = "\033[1m"; CYAN = "\033[96m"

def print_step(n: int, text: str):
    filled = "█" * n + "░" * (len(STEPS) - n)
    print(f"\n{CYAN}{B}  [{n}/{len(STEPS)}] {filled}  {text}{R}")
    time.sleep(0.2)

def print_ok(msg):   print(f"  \033[92m✔  {msg}{R}")
def print_warn(msg): print(f"  \033[93m⚠  {msg}{R}")
def print_info(msg): print(f"  \033[96mℹ  {msg}{R}")

def print_report(detections: list[dict]):
    print(f"\n{B}{'═'*64}{R}")
    print(f"{B}   🏥  SURGERY ROOM WASTE CLASSIFICATION REPORT{R}")
    print(f"{B}{'═'*64}{R}")
    print(f"  {B}{'OBJECT':<28} {'CONF':>5}   {'BIN':<10} WASTE TYPE{R}")
    print(f"  {'─'*27} {'─'*5}   {'─'*9} {'─'*25}")
    from collections import defaultdict
    grouped = defaultdict(list)
    for d in detections:
        ansi = BINS[d['bin']]['ansi']
        print(f"  {d['label'].title():<28} {d['conf']*100:>4.0f}%"
              f"   {ansi}{d['bin']:<10}{R} {BINS.get(d['bin'],{}).get('label','')}")
        grouped[d['bin']].append(d['label'].title())
    print(f"\n{B}  Disposal Summary:{R}")
    for bin_name, items in sorted(grouped.items()):
        ansi = BINS[bin_name]['ansi']
        meta = BINS[bin_name]
        print(f"  {ansi}{B}{meta['emoji']} {bin_name}{R} — {meta['label']}")
        for it in items:
            print(f"       • {it}")
    print()


# ─────────────────────────────────────────────
#  LED SIMULATION
# ─────────────────────────────────────────────
def draw_led_panel(frame: np.ndarray, active: str | None) -> np.ndarray:
    h, w   = frame.shape[:2]
    ph     = 80
    panel  = np.full((ph, w, 3), (25, 25, 25), dtype=np.uint8)
    n      = len(BINS)
    gap    = w // (n + 1)

    cv2.putText(panel, "LED INDICATOR", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160,160,160), 1)

    for i, (bin_name, meta) in enumerate(BINS.items()):
        cx  = gap * (i + 1)
        cy  = ph // 2 + 8
        bgr = meta["bgr"]

        if bin_name == active:
            glow = np.zeros_like(panel)
            cv2.circle(glow,  (cx, cy), 30, bgr, -1)
            panel = cv2.addWeighted(panel, 1.0, glow, 0.4, 0)
            cv2.circle(panel, (cx, cy), 22, bgr, -1)
            cv2.circle(panel, (cx, cy), 24, (255,255,255), 1)
            txt_c = (255, 255, 255)
        else:
            dim = tuple(max(0, c // 6) for c in bgr)
            cv2.circle(panel, (cx, cy), 16, dim, -1)
            txt_c = (70, 70, 70)

        cv2.putText(panel, bin_name, (cx - 22, cy + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, txt_c, 1, cv2.LINE_AA)

    return np.vstack([panel, frame])


# ─────────────────────────────────────────────
#  STEP SIDEBAR
# ─────────────────────────────────────────────
def draw_step_sidebar(frame: np.ndarray, current: int) -> np.ndarray:
    h, w    = frame.shape[:2]
    sw      = 250
    sidebar = np.full((h, sw, 3), (18, 18, 18), dtype=np.uint8)

    cv2.putText(sidebar, "OR WASTE PIPELINE", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190,190,190), 1)
    cv2.line(sidebar, (8, 36), (sw-8, 36), (55,55,55), 1)

    step_h = (h - 48) // len(STEPS)
    for i, text in enumerate(STEPS, 1):
        y = 48 + (i-1)*step_h + step_h//2
        if   i < current:  col = (0,160,0);   sym = "✔"
        elif i == current: col = (0,210,230);  sym = "▶"
        else:              col = (55,55,55);   sym = "○"

        if i < len(STEPS):
            cv2.line(sidebar, (22,y+10),(22,y+step_h),(50,50,50),1)

        cv2.circle(sidebar, (22, y), 8,
                   col if i <= current else (45,45,45), -1)

        words = text.split()
        line1 = " ".join(words[:3])
        line2 = " ".join(words[3:])
        cv2.putText(sidebar, f"{sym} {line1}", (36, y+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        if line2:
            cv2.putText(sidebar, f"  {line2}", (36, y+17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    return np.hstack([frame, sidebar])


# ─────────────────────────────────────────────
#  BOUNDING BOXES
# ─────────────────────────────────────────────
def draw_boxes(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        bgr = BINS.get(d["bin"],{}).get("bgr",(120,120,120))
        txt = f"{d['label'].title()} → {d['bin']} ({d['conf']*100:.0f}%)"
        cv2.rectangle(out,(x1,y1),(x2,y2),bgr,2)
        (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
        cv2.rectangle(out,(x1,y1-th-8),(x1+tw+4,y1),bgr,-1)
        cv2.putText(out,txt,(x1+2,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────
#  COMBINED DISPLAY FRAME
# ─────────────────────────────────────────────
def build_display(frame, step, dets=[], active=None):
    return draw_step_sidebar(
        draw_led_panel(draw_boxes(frame, dets), active),
        step
    )


# ─────────────────────────────────────────────
#  SOURCE INDICATOR OVERLAY  (shows IP cam info)
# ─────────────────────────────────────────────
def draw_source_badge(frame: np.ndarray, source_label: str) -> np.ndarray:
    """Draw a small source badge (e.g. 'DroidCam  192.168.1.42:4747') on frame."""
    out = frame.copy()
    txt = f"CAM: {source_label}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(out, (6, 6), (tw + 14, th + 14), (20, 20, 20), -1)
    cv2.putText(out, txt, (10, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 230), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(frame: np.ndarray, model: YOLO,
                 conf: float, iou: float, win: str) -> None:

    def show(f, step, dets=[], active=None, wait=1):
        cv2.imshow(win, build_display(f, step, dets, active))
        cv2.waitKey(wait)

    print_step(2, STEPS[1])
    ts = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(OUTPUT_DIR / f"capture_{ts}.jpg"), frame)
    print_ok("Frame saved")
    show(frame, 2); time.sleep(0.5)

    print_step(3, STEPS[2])
    show(frame, 3)
    t0    = time.perf_counter()
    dets  = multi_pass_detect(model, frame, conf, iou)
    ms    = (time.perf_counter()-t0)*1000
    print_ok(f"Inference: {ms:.0f} ms  ({len(dets)} object(s) detected)")
    for d in dets:
        print_ok(f"  {d['label'].title():<30} → {d['bin']}  ({d['conf']*100:.0f}%)")

    if not dets:
        print_warn("Nothing detected — improve lighting or angle.")
        time.sleep(1.5); return

    show(frame, 3, dets); time.sleep(0.8)

    print_step(4, STEPS[3])
    primary = max(dets, key=lambda d: d["conf"])
    meta    = BINS[primary["bin"]]
    print_ok(f"Primary  : {primary['label'].title()}")
    print_ok(f"Bin      : {meta['emoji']} {primary['bin']} — {meta['label']}")
    show(frame, 4, dets); time.sleep(0.8)

    print_step(5, STEPS[4])
    print_ok(f"LED ON → {meta['emoji']} {primary['bin']}")
    for _ in range(3):
        show(frame, 5, dets, primary["bin"]); cv2.waitKey(350)
        show(frame, 5, dets, None);           cv2.waitKey(200)
    show(frame, 5, dets, primary["bin"]); time.sleep(1.0)

    print_step(6, STEPS[5])
    out_path = str(OUTPUT_DIR / f"result_{ts}.jpg")
    cv2.imwrite(out_path, draw_boxes(frame, dets))
    print_ok(f"Result saved → {out_path}")
    print_report(dets)
    show(frame, 6, dets, primary["bin"]); time.sleep(2.5)


# ─────────────────────────────────────────────
#  ENTRY POINTS
# ─────────────────────────────────────────────
WIN = "Smart OR Waste Segregation PoC  (SPACE=classify  ESC=quit)"

def image_mode(path, model, conf, iou):
    frame = cv2.imread(path)
    if frame is None: sys.exit(f"❌  Cannot read: {path}")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print_step(1, STEPS[0] + f"  [{path}]")
    cv2.imshow(WIN, build_display(frame, 1)); cv2.waitKey(600)
    run_pipeline(frame, model, conf, iou, WIN)
    print(f"\n  Press any key to close …")
    cv2.waitKey(0); cv2.destroyAllWindows()


def webcam_mode(model, conf, iou, source: str | int = 0,
                source_label: str = "local"):
    """
    Live camera loop.  Works with:
      • local webcam (int index)
      • DroidCam over USB/Wi-Fi  (HTTP MJPEG)
      • IP Webcam Android app    (HTTP MJPEG)
      • Any RTSP stream          (H.264)
    """
    cap = open_camera(source)
    verify_camera(cap, source)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print_step(1, STEPS[0])
    print(f"\n  {B}SPACE{R} = classify     {B}ESC{R} = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            # IP streams occasionally drop a frame — retry once
            ret, frame = cap.read()
            if not ret:
                print_warn("Lost connection to camera stream — reconnecting…")
                cap.release()
                time.sleep(2)
                cap = open_camera(source)
                if not cap.isOpened():
                    sys.exit("❌  Could not reconnect to camera.")
                continue

        display = draw_source_badge(frame, source_label)
        cv2.imshow(WIN, build_display(display, 1))
        key = cv2.waitKey(1) & 0xFF

        if key == 32:   # SPACE
            run_pipeline(frame, model, conf, iou, WIN)
            print_step(1, STEPS[0])
            print(f"  {B}Ready — show next item …{R}\n")
        elif key == 27: # ESC
            break

    cap.release(); cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description="Smart OR Waste Segregation — Surgery Room PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Camera source examples
──────────────────────
  Local webcam (default)     : (no flag needed, or --webcam 0)
  DroidCam Wi-Fi             : --droidcam 192.168.1.42
  DroidCam custom port       : --droidcam 192.168.1.42:4747
  IP Webcam Android app      : --droidcam http://192.168.1.42:8080/videofeed
  RTSP stream                : --droidcam rtsp://192.168.1.42:4747/h264_pcm.sdp
  EpocCam                    : --droidcam http://192.168.1.42:2431/video
  Any MJPEG URL              : --droidcam http://<ip>:<port>/video
        """,
    )
    ap.add_argument("--image",    metavar="PATH",  help="Analyse a still image")
    ap.add_argument("--webcam",   metavar="INDEX", default="0",
                    help="Local webcam device index (default: 0)")
    ap.add_argument("--droidcam", metavar="IP[:PORT] | URL",
                    help="DroidCam / IP Webcam address (see examples below)")
    ap.add_argument("--conf",  type=float, default=CONF,
                    help=f"Detection confidence (default {CONF})")
    ap.add_argument("--iou",   type=float, default=IOU,
                    help=f"NMS IoU threshold (default {IOU})")
    ap.add_argument("--model", default=MODEL,
                    help=f"YOLOv8 weights (default: {MODEL})")
    args = ap.parse_args()

    print(f"\n{B}{'═'*58}{R}")
    print(f"{B}  🏥  Smart OR Waste Segregation — Surgery Room PoC{R}")
    print(f"{B}{'═'*58}{R}")
    print(f"\n  Model : {args.model}")
    print(f"  Conf  : {args.conf}   IOU : {args.iou}   ImgSize : {IMG_SIZE}")
    print(f"  Items in waste map : {len([v for v in WASTE_MAP.values() if v])}")

    if args.droidcam:
        source       = resolve_camera_source(args.droidcam)
        source_label = args.droidcam
    else:
        source       = resolve_camera_source(args.webcam)
        source_label = f"local:{args.webcam}"

    print(f"  Camera: {source_label}")
    print(f"  Loading model …")
    model = YOLO(args.model)
    print(f"  Model ready.\n")

    if args.image:
        image_mode(args.image, model, args.conf, args.iou)
    else:
        webcam_mode(model, args.conf, args.iou,
                    source=source, source_label=source_label)


if __name__ == "__main__":
    main()
