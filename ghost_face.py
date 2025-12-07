"""
realtime_multiface_swap.py

Realtime multi-face swap (single webcam) using MediaPipe FaceMesh + OpenCV.

Controls:
 - 1..9 : select corresponding target face from faces/ (1 -> index 0, 2 -> index 1, ...)
 - a    : add current face as a new target (saved into faces/)
 - s    : toggle swap on/off
 - q / Esc : quit

Requirements:
 pip install opencv-python mediapipe numpy

Notes:
 - Put initial target images into a ./faces/ directory or press 'a' to capture.
 - This uses triangulation computed on each target when loaded to avoid recomputing each frame.
"""

import cv2
import os
import time
import glob
import numpy as np
import mediapipe as mp
from datetime import datetime

FACES_DIR = "faces"
MAX_KEY_TARGETS = 9  # keys 1..9

# ----------------- Utility functions -----------------
mp_face_mesh = mp.solutions.face_mesh

def ensure_faces_dir():
    if not os.path.exists(FACES_DIR):
        os.makedirs(FACES_DIR)

def list_face_files():
    patterns = [os.path.join(FACES_DIR, "*.jpg"), os.path.join(FACES_DIR, "*.png"), os.path.join(FACES_DIR, "*.jpeg")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files.sort()
    return files

def get_landmarks_mediapipe(img, face_mesh, static=False):
    # returns list of (x,y) pixel coords or None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if static:
        res = face_mesh.process(img_rgb)
    else:
        res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    return pts

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def compute_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(tuple(p))
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    # For robust index finding, allow small tolerance
    pts_arr = np.array(points)
    for t in triangleList:
        pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
        indices = []
        for p in pts:
            # find nearest point index
            dist = np.sum((pts_arr - p) ** 2, axis=1)
            idx = np.argmin(dist)
            if dist[idx] < 25:  # squared tolerance ~5 px. Adjust if needed
                indices.append(int(idx))
        if len(indices) == 3:
            # dedupe triangle
            tri = tuple(indices)
            if tri not in delaunayTri:
                delaunayTri.append(tri)
    return delaunayTri

# ----------------- TargetFace class -----------------
class TargetFace:
    def __init__(self, img_path=None, image=None, idx=0):
        self.idx = idx
        if img_path:
            self.path = img_path
            self.image = cv2.imread(img_path)
        else:
            self.path = None
            self.image = image
        if self.image is None:
            raise ValueError("Target image couldn't be loaded.")
        self.h, self.w = self.image.shape[:2]
        # landmarks and triangulation will be filled after detect_landmarks
        self.landmarks = None  # list of (x,y)
        self.delaunay = None
        self.mask = None
        self.hull = None

    def detect_landmarks_and_prepare(self, face_mesh):
        lm = get_landmarks_mediapipe(self.image, face_mesh, static=True)
        if lm is None:
            return False
        self.landmarks = np.array(lm, dtype=np.int32)
        rect = (0, 0, self.w, self.h)
        self.delaunay = compute_delaunay_triangles(rect, self.landmarks.tolist())
        # mask & hull
        hull = cv2.convexHull(self.landmarks)
        self.hull = hull
        m = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.fillConvexPoly(m, hull, 255)
        self.mask = m
        return True

# ----------------- FaceSwapEngine -----------------
class FaceSwapEngine:
    def __init__(self, scale_for_speed=0.6):
        self.scale = scale_for_speed  # process at scaled resolution for speed
        ensure_faces_dir()
        self.targets = []  # list of TargetFace
        self.current_target_idx = None
        self.swap_enabled = True
        self.face_mesh_static = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                      refine_landmarks=True, min_detection_confidence=0.5)
        self.face_mesh_live = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                    refine_landmarks=True, min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)
        self.load_targets_from_folder()

    def load_targets_from_folder(self):
        self.targets = []
        files = list_face_files()
        for i, f in enumerate(files):
            try:
                t = TargetFace(img_path=f, idx=i)
                ok = t.detect_landmarks_and_prepare(self.face_mesh_static)
                if ok:
                    self.targets.append(t)
                else:
                    print(f"[WARN] No face landmarks in {f}; skipping.")
            except Exception as e:
                print(f"[ERROR] Loading {f}: {e}")
        if len(self.targets) > 0:
            self.current_target_idx = 0
        else:
            self.current_target_idx = None
        print(f"[INFO] Loaded {len(self.targets)} target(s).")

    def add_target_from_frame(self, frame):
        # Detect face in original frame, crop tight box using landmarks, save aligned small image into faces/
        lm = get_landmarks_mediapipe(frame, self.face_mesh_static, static=True)
        if lm is None:
            print("[INFO] No face detected; cannot add.")
            return False
        pts = np.array(lm, dtype=np.int32)
        x_min = max(0, pts[:,0].min() - 20)
        y_min = max(0, pts[:,1].min() - 20)
        x_max = min(frame.shape[1], pts[:,0].max() + 20)
        y_max = min(frame.shape[0], pts[:,1].max() + 20)
        crop = frame[y_min:y_max, x_min:x_max].copy()
        # save with timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(FACES_DIR, f"face_{ts}.jpg")
        cv2.imwrite(filename, crop)
        print(f"[INFO] Saved new target face: {filename}")
        # load into engine
        try:
            t = TargetFace(img_path=filename, idx=len(self.targets))
            if t.detect_landmarks_and_prepare(self.face_mesh_static):
                self.targets.append(t)
                self.current_target_idx = len(self.targets) - 1
                print(f"[INFO] Added as target index {self.current_target_idx}")
                return True
            else:
                print("[WARN] Saved image doesn't contain detectable landmarks.")
                return False
        except Exception as e:
            print("[ERROR] Adding target failed:", e)
            return False

    def select_target_by_key(self, key):
        # map '1' -> 0, '2'->1, etc.
        idx = key - ord('1')  # '1' ascii to 0
        if 0 <= idx < len(self.targets):
            self.current_target_idx = idx
            print(f"[INFO] Selected target {idx}")
            return True
        return False

    def process_frame(self, frame_original):
        # Resize for speed
        h0, w0 = frame_original.shape[:2]
        w = int(w0 * self.scale)
        h = int(h0 * self.scale)
        frame = cv2.resize(frame_original, (w, h))
        lm = get_landmarks_mediapipe(frame, self.face_mesh_live, static=False)
        if lm is None:
            return frame_original  # nothing to do

        if self.current_target_idx is None or not self.swap_enabled:
            # show label of no-swap
            out = frame_original.copy()
            cv2.putText(out, "Swap OFF" if not self.swap_enabled else "No target loaded", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            return out

        target = self.targets[self.current_target_idx]
        # we need to map scaled live landmarks to original frame coordinates:
        scale_x = w0 / w
        scale_y = h0 / h
        dst_pts = np.array([(int(x*scale_x), int(y*scale_y)) for (x,y) in lm], dtype=np.int32)
        src_pts = target.landmarks  # in target image coords

        # Prepare warped target sized to original frame
        warped_target = np.zeros_like(frame_original)

        # For each triangle in target.delaunay, warp to dst
        for tri in target.delaunay:
            x, y, z = tri
            t_tri = [src_pts[x], src_pts[y], src_pts[z]]
            f_tri = [dst_pts[x], dst_pts[y], dst_pts[z]]

            # bounding rects in target and frame
            t_rect = cv2.boundingRect(np.float32([t_tri]))
            f_rect = cv2.boundingRect(np.float32([f_tri]))

            # offsets
            t_tri_offset = []
            f_tri_offset = []
            for i_pt in range(3):
                t_tri_offset.append(((t_tri[i_pt][0] - t_rect[0]), (t_tri[i_pt][1] - t_rect[1])))
                f_tri_offset.append(((f_tri[i_pt][0] - f_rect[0]), (f_tri[i_pt][1] - f_rect[1])))

            # crop target triangle
            t_cropped = target.image[t_rect[1]:t_rect[1]+t_rect[3], t_rect[0]:t_rect[0]+t_rect[2]]
            if t_cropped.size == 0 or f_rect[2] <= 0 or f_rect[3] <= 0:
                continue

            warped = apply_affine_transform(t_cropped, t_tri_offset, f_tri_offset, (f_rect[2], f_rect[3]))

            # mask and paste
            mask = np.zeros((f_rect[3], f_rect[2], 3), dtype=np.uint8)
            points = np.array([[f_tri_offset[0][0], f_tri_offset[0][1]],
                               [f_tri_offset[1][0], f_tri_offset[1][1]],
                               [f_tri_offset[2][0], f_tri_offset[2][1]]], dtype=np.int32)
            cv2.fillConvexPoly(mask, points, (1,1,1), 16, 0)

            # --- SAFE PASTE WITH CLAMPING ---
            x, y, w_t, h_t = f_rect
            H, W = warped_target.shape[:2]

# clamp coords inside frame
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(W, x + w_t)
            y2 = min(H, y + h_t)

            paste_w = x2 - x1
            paste_h = y2 - y1

# skip invalid region
            if paste_w <= 0 or paste_h <= 0:
              continue

# resize warped + mask exactly to paste region
            warped_resized = cv2.resize(warped, (paste_w, paste_h))
            mask_resized = cv2.resize(mask, (paste_w, paste_h))

# paste
            region = warped_target[y1:y2, x1:x2]
            warped_target[y1:y2, x1:x2] = region * (1 - mask_resized) + warped_resized * mask_resized


        # Create mask for destination face (from dst_pts)
        hull2 = cv2.convexHull(dst_pts)
        mask = np.zeros((h0, w0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull2, 255)
        center = (int((hull2[:,0,0].min() + hull2[:,0,0].max())/2),
                  int((hull2[:,0,1].min() + hull2[:,0,1].max())/2))

        # Seamless clone
        try:
            output = cv2.seamlessClone(warped_target, frame_original, mask, center, cv2.NORMAL_CLONE)
        except Exception as e:
            print("[WARN] seamlessClone failed:", e)
            output = frame_original.copy()

        # overlay small preview of selected target
        preview_size = 160
        tgt_small = cv2.resize(target.image, (preview_size, preview_size))
        x_off = 10
        y_off = 10
        output[y_off:y_off+preview_size, x_off:x_off+preview_size] = tgt_small

        # draw overlays
        label = f"Target #{self.current_target_idx+1} - {os.path.basename(target.path) if target.path else 'captured'}"
        cv2.putText(output, label, (10, preview_size + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(output, "Press 'a' to add face | 's' toggle swap | 1-9 select", (10, preview_size + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

        return output

# ----------------- Main loop -----------------
def main():
    engine = FaceSwapEngine(scale_for_speed=0.6)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not opened.")
        return

    print("[INFO] Controls: 1..9 select target, 'a' add face, 's' toggle swap, 'q' or Esc quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = engine.process_frame(frame)
        cv2.imshow("Realtime Multi-FaceSwap", out)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif ord('1') <= key <= ord('9'):
            engine.select_target_by_key(key)
        elif key == ord('a'):
            # add new face (use original resolution frame)
            added = engine.add_target_from_frame(frame)
            if not added:
                print("[INFO] Could not add face.")
        elif key == ord('s'):
            engine.swap_enabled = not engine.swap_enabled
            print(f"[INFO] Swap enabled: {engine.swap_enabled}")
        elif key == ord('r'):
            # reload targets from folder (useful if you manually add files)
            engine.load_targets_from_folder()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
