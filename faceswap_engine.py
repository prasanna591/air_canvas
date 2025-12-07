"""
realtime_multiface_swap_upgraded.py

Upgraded realtime multi-face swap (single webcam) using MediaPipe FaceMesh + OpenCV.

Controls:
 - 1..9 : select corresponding target face from faces/ (1 -> index 0, 2 -> index 1, ...)
 - a    : add current face as a new target (saved into faces/)
 - s    : toggle swap on/off
 - q / Esc : quit

Requirements:
 pip install opencv-python mediapipe numpy
"""

import cv2
import os
import glob
import numpy as np
import mediapipe as mp
from datetime import datetime

FACES_DIR = "faces"
MAX_KEY_TARGETS = 9

mp_face_mesh = mp.solutions.face_mesh

# ---------- Utilities ----------
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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        # Subdiv2D requires points strictly inside rect, clamp if necessary
        x = min(max(p[0], rect[0]+1), rect[2]-1)
        y = min(max(p[1], rect[1]+1), rect[3]-1)
        subdiv.insert((int(x), int(y)))
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    pts_arr = np.array(points)
    for t in triangleList:
        pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
        indices = []
        for p in pts:
            dist = np.sum((pts_arr - p) ** 2, axis=1)
            idx = np.argmin(dist)
            if dist[idx] < 100:  # squared tolerance (increase if faces differ in scale)
                indices.append(int(idx))
        if len(indices) == 3:
            tri = tuple(indices)
            # avoid duplicates (unordered)
            if tri not in delaunayTri:
                delaunayTri.append(tri)
    return delaunayTri

def color_correct(src_crop, dst_crop, mask_crop):
    """
    Adjust src_crop colors to match dst_crop inside mask_crop.
    Uses per-channel mean/std matching in float space.
    """
    src = src_crop.astype(np.float32)
    dst = dst_crop.astype(np.float32)
    mask_bool = (mask_crop > 0)
    if mask_bool.sum() < 10:
        return src_crop
    for c in range(3):
        s_vals = src[:,:,c][mask_bool]
        d_vals = dst[:,:,c][mask_bool]
        s_mean, s_std = s_vals.mean(), s_vals.std() if s_vals.std() > 1e-2 else 1.0
        d_mean, d_std = d_vals.mean(), d_vals.std() if d_vals.std() > 1e-2 else 1.0
        # match mean and std
        src[:,:,c] = (src[:,:,c] - s_mean) * (d_std / s_std) + d_mean
    np.clip(src, 0, 255, out=src)
    return src.astype(np.uint8)

# ---------- TargetFace ----------
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
        self.landmarks = None
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
        hull = cv2.convexHull(self.landmarks)
        self.hull = hull
        m = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.fillConvexPoly(m, hull, 255)
        self.mask = m
        return True

# ---------- Engine ----------
class FaceSwapEngine:
    def __init__(self, scale_for_speed=0.6, use_seamless_clone=False):
        self.scale = scale_for_speed
        self.use_seamless_clone = use_seamless_clone
        ensure_faces_dir()
        self.targets = []
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
                    print(f"[WARN] No landmarks in {f}; skipping.")
            except Exception as e:
                print(f"[ERROR] Loading {f}: {e}")
        self.current_target_idx = 0 if len(self.targets) > 0 else None
        print(f"[INFO] Loaded {len(self.targets)} target(s).")

    def add_target_from_frame(self, frame):
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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(FACES_DIR, f"face_{ts}.jpg")
        cv2.imwrite(filename, crop)
        print(f"[INFO] Saved new target face: {filename}")
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
        idx = key - ord('1')
        if 0 <= idx < len(self.targets):
            self.current_target_idx = idx
            print(f"[INFO] Selected target {idx}")
            return True
        return False

    def process_frame(self, frame_original):
        h0, w0 = frame_original.shape[:2]
        w = int(w0 * self.scale)
        h = int(h0 * self.scale)
        # small processing frame for detection (faster)
        small = cv2.resize(frame_original, (w, h))
        lm = get_landmarks_mediapipe(small, self.face_mesh_live, static=False)
        if lm is None:
            return frame_original

        if self.current_target_idx is None or not self.swap_enabled:
            out = frame_original.copy()
            cv2.putText(out, "Swap OFF" if not self.swap_enabled else "No target loaded", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            return out

        target = self.targets[self.current_target_idx]

        # map small->original coords
        scale_x = w0 / w
        scale_y = h0 / h
        dst_pts = np.array([(int(x*scale_x), int(y*scale_y)) for (x,y) in lm], dtype=np.int32)
        src_pts = target.landmarks

        warped_target = np.zeros_like(frame_original)

        # warp triangles from target -> destination
        for tri in target.delaunay:
            x, y, z = tri
            t_tri = [src_pts[x], src_pts[y], src_pts[z]]
            f_tri = [dst_pts[x], dst_pts[y], dst_pts[z]]

            t_rect = cv2.boundingRect(np.float32([t_tri]))
            f_rect = cv2.boundingRect(np.float32([f_tri]))

            # skip degenerate rects
            if t_rect[2] <= 0 or t_rect[3] <= 0 or f_rect[2] <= 0 or f_rect[3] <= 0:
                continue

            t_tri_offset = []
            f_tri_offset = []
            for i_pt in range(3):
                t_tri_offset.append(((t_tri[i_pt][0] - t_rect[0]), (t_tri[i_pt][1] - t_rect[1])))
                f_tri_offset.append(((f_tri[i_pt][0] - f_rect[0]), (f_tri[i_pt][1] - f_rect[1])))

            # crop source triangle
            t_cropped = target.image[t_rect[1]:t_rect[1]+t_rect[3], t_rect[0]:t_rect[0]+t_rect[2]]
            if t_cropped.size == 0:
                continue

            try:
                warped = apply_affine_transform(t_cropped, t_tri_offset, f_tri_offset, (f_rect[2], f_rect[3]))
            except Exception as e:
                # skip problematic triangle
                # print("[DEBUG] warp failed:", e)
                continue

            mask = np.zeros((f_rect[3], f_rect[2], 3), dtype=np.uint8)
            points = np.array([[f_tri_offset[0][0], f_tri_offset[0][1]],
                               [f_tri_offset[1][0], f_tri_offset[1][1]],
                               [f_tri_offset[2][0], f_tri_offset[2][1]]], dtype=np.int32)
            cv2.fillConvexPoly(mask, points, (1,1,1), 16, 0)

            dest_roi = warped_target[f_rect[1]:f_rect[1]+f_rect[3], f_rect[0]:f_rect[0]+f_rect[2]]
            # ensure shapes match
            if dest_roi.shape[:2] != warped.shape[:2]:
                # skip if mismatch (safety)
                continue

            # paste using mask (1 or 0)
            warped_target[f_rect[1]:f_rect[1]+f_rect[3], f_rect[0]:f_rect[0]+f_rect[2]] = \
                dest_roi * (1 - mask) + warped * mask

        # build full-face mask from destination hull
        hull2 = cv2.convexHull(dst_pts)
        mask_full = np.zeros((h0, w0), dtype=np.uint8)
        cv2.fillConvexPoly(mask_full, hull2, 255)

        # If too small face detected, skip
        if mask_full.sum() < 1000:
            return frame_original

        # bounding rect of face region
        x,y,wf,hf = cv2.boundingRect(hull2)
        # crop both images
        warped_crop = warped_target[y:y+hf, x:x+wf]
        frame_crop = frame_original[y:y+hf, x:x+wf]
        mask_crop = mask_full[y:y+hf, x:x+wf]

        if warped_crop.size == 0 or frame_crop.size == 0 or mask_crop.size == 0:
            return frame_original

        # Color-correct warped_crop to match frame_crop (inside mask)
        try:
            corrected = color_correct(warped_crop, frame_crop, mask_crop)
        except Exception as e:
            print("[WARN] color correction failed:", e)
            corrected = warped_crop

        # feather mask for smooth blending
        mask_feather = cv2.GaussianBlur(mask_crop, (31,31), 0).astype(np.float32)/255.0
        mask_feather = np.repeat(mask_feather[:, :, None], 3, axis=2)

        # alpha blend corrected onto frame_crop
        blended_crop = (corrected.astype(np.float32) * mask_feather + frame_crop.astype(np.float32) * (1.0 - mask_feather)).astype(np.uint8)

        # Create output and put blended region back
        output = frame_original.copy()
        output[y:y+hf, x:x+wf] = blended_crop

        # As a final optional refinement, you can use seamlessClone with the binary mask
        if self.use_seamless_clone:
            try:
                center = (int(x + wf/2), int(y + hf/2))
                output = cv2.seamlessClone(output, frame_original, mask_full, center, cv2.NORMAL_CLONE)
            except Exception as e:
                print("[WARN] seamlessClone failed (continuing without):", e)
                # keep blended 'output'

        # overlay preview
        preview_size = 160
        tgt_small = cv2.resize(target.image, (preview_size, preview_size))
        x_off = 10
        y_off = 10
        output[y_off:y_off+preview_size, x_off:x_off+preview_size] = tgt_small

        label = f"Target #{self.current_target_idx+1} - {os.path.basename(target.path) if target.path else 'captured'}"
        cv2.putText(output, label, (10, preview_size + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(output, "Press 'a' to add face | 's' toggle swap | 1-9 select", (10, preview_size + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

        return output

# ---------- Main ----------
def main():
    engine = FaceSwapEngine(scale_for_speed=0.6, use_seamless_clone=False)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not opened.")
        return

    print("[INFO] Loaded {} target(s). Controls: 1..9 select target, 'a' add face, 's' toggle swap, 'r' reload, 'q'/Esc quit.".format(len(engine.targets)))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = engine.process_frame(frame)
        cv2.imshow("Realtime Multi-FaceSwap (upgraded)", out)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif ord('1') <= key <= ord('9'):
            engine.select_target_by_key(key)
        elif key == ord('a'):
            added = engine.add_target_from_frame(frame)
            if not added:
                print("[INFO] Could not add face.")
        elif key == ord('s'):
            engine.swap_enabled = not engine.swap_enabled
            print(f"[INFO] Swap enabled: {engine.swap_enabled}")
        elif key == ord('r'):
            engine.load_targets_from_folder()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
