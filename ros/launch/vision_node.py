#!/usr/bin/env python3
"""Nodo visione: attende /home_reached, scatta la foto, esegue la rete e
pubblica i punti (JSON in std_msgs/String) su /sweater/seams.

Va lanciato con l'interprete del venv:
    <rete>/.venv/bin/python -m coordinate_converter.vision_node
"""
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

PKG_DIR = os.path.dirname(os.path.realpath(__file__))
RETE_DIR = os.path.join(PKG_DIR, "rete")
for _p in (RETE_DIR, os.path.join(RETE_DIR, "codice_train_test")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from acquisizione import Camera              # noqa: E402
from yolo_inference import SweaterDetector   # noqa: E402

DEFAULT_WEIGHTS = os.path.join(RETE_DIR, "sweater_segmentation", "yolo_run",
                               "weights", "best.pt")

LATCHED = QoSProfile(depth=1,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.declare_parameter("weights_path", DEFAULT_WEIGHTS)
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("settle_time", 2.0)  # s di attesa robot fermo

        self.declare_parameter("rgb_path", "")     # se impostato: test senza camera
        self.declare_parameter("depth_path", "")   # opzionale, default: stesso nome con _depth.png

        weights = self.get_parameter("weights_path").value
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Pesi non trovati: {weights}")

        self.camera = Camera()  # non apre la camera: lo fa capture()
        # modello caricato UNA volta; intrinseci 1280x720 come in verify_depth_extraction
        self.detector = SweaterDetector(
            weights, fx=908.89948, fy=908.29236, ppx=647.69177, ppy=356.04810)

        self.done = False
        self.pub = self.create_publisher(String, "/sweater/seams", LATCHED)
        self.create_subscription(Bool, "/home_reached", self.on_home, LATCHED)
        self.get_logger().info("vision_node pronto, in attesa di home_reached")

    def on_home(self, msg):
        if self.done:
            return
        if not msg.data:
            self.get_logger().error("Home non raggiunta: nessuna acquisizione")
            return
        self.done = True

        time.sleep(self.get_parameter("settle_time").value)  # robot fermo
        try:
            rgb_test = self.get_parameter("rgb_path").value
            if rgb_test:
                shot = self.load_test_shot(rgb_test, self.get_parameter("depth_path").value)
            else:
                shot = self.camera.capture()
            intr = shot["intrinsics"]
            self.detector.fx, self.detector.fy = intr["fx"], intr["fy"]
            self.detector.ppx, self.detector.ppy = intr["ppx"], intr["ppy"]

            coords, _ = self.detector.process_image(
                shot["rgb_path"], depth_path=shot["depth_path"],
                conf_threshold=self.get_parameter("conf_threshold").value)
            payload = self.build_payload(coords, shot)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Errore visione: {e}")
            payload = {"success": False, "message": str(e)}

        payload["stamp"] = time.time()
        self.pub.publish(String(data=json.dumps(payload)))
        if payload["success"]:
            self.get_logger().info("Punti pubblicati su /sweater/seams")

    def load_test_shot(self, rgb_path, depth_path):
        import cv2
        if not depth_path:
            depth_path = rgb_path.replace("_rgb.png", "_depth.png")
        img = cv2.imread(rgb_path)
        if img is None:
            raise FileNotFoundError(f"RGB non leggibile: {rgb_path}")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth non trovata: {depth_path}")
        # stessi intrinseci usati in verify_depth_extraction, in base alla risoluzione
        if img.shape[1] == 1280:
            intr = dict(fx=908.89948, fy=908.29236, ppx=647.69177, ppy=356.04810)
        else:
            intr = dict(fx=605.93298, fy=605.52826, ppx=325.12787, ppy=237.36540)
        self.get_logger().info(f"Modalità test: {rgb_path}")
        return dict(rgb_path=rgb_path, depth_path=depth_path, intrinsics=intr)

    def build_payload(self, coords, shot):
        # polsini ordinati per X (camera): primo = sx, secondo = dx
        cuffs = sorted(coords.get("Cuff", []), key=lambda c: c["start"][0])
        hems = coords.get("Hem", [])
        collars = coords.get("Collar", [])

        if len(cuffs) != 2 or not hems or not collars:
            msg = (f"Rilevamento incompleto: Cuff={len(cuffs)}, "
                   f"Hem={len(hems)}, Collar={len(collars)}")
            self.get_logger().error(msg)
            return {"success": False, "message": msg}

        # punti in mm nel frame camera, come si aspetta rete_to_base_tf
        return {
            "success": True,
            "message": "ok",
            "unit": "mm",
            "polsino_sx": [cuffs[0]["start"], cuffs[0]["end"]],
            "polsino_dx": [cuffs[1]["start"], cuffs[1]["end"]],
            "fondo_maglia": hems[0]["points"],
            "colletto": collars[0]["curve_points"],
            "rgb_path": shot["rgb_path"],
            "depth_path": shot["depth_path"],
        }


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()