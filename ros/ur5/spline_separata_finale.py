#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformListener
from builtin_interfaces.msg import Duration

from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance


class KinematicController(Node):

    def __init__(self):
        super().__init__("kinematic_controller")

        # Parametri DH UR5
        self.a     = [0,       -0.425,   -0.39225, 0,       0,       0      ]
        self.alpha = [np.pi/2,  0,        0,        np.pi/2, -np.pi/2, 0     ]
        self.d     = [0.089159, 0,        0,        0.10915, 0.09465, 0.0823 ]

        self.q_min = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.q_max = np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])
        self.dq_max = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28])

        self.Kp = 20#14.5
        self.Ko = 8.0#5.0
        self.K  = np.diag([self.Kp, self.Kp, self.Kp,
                           self.Ko, self.Ko, self.Ko])
        self.lambda_dls = 0.06 #prima 0.03

        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # Publisher aggiornato
        self.pub_joint = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )

        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        #UFFICIALE
        """self.pub_joint = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            10
        )"""
        self.pub_trajectory_desired = self.create_publisher(
            PoseArray, '/trajectory_desired', 10
        )

        # Subscriber
        self.q_current = None
        self.sub_joint = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # TF buffer + listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = None
        self.idx   = 0
        self.dt    = 0.01#0.025#0.04#0.025  #0.05 corretto
        self.finito_traiettoria = False

        self.get_logger().info("KinematicController avviato!")


    def joint_state_callback(self, msg: JointState):
        q = np.zeros(6)
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]
            else:
                self.get_logger().warn(f"Giunto {name} non trovato!")
                return

        self.q_current = q

        if not self.finito_traiettoria:
            if not self.tf_buffer.can_transform('base', 'camera_link', rclpy.time.Time()):
                self.get_logger().warn("TF non ancora pronto, aspetto...")
                return
            self.genera_traiettoria_diversa()

        """if self.timer is None and self.q_current is not None and self.finito_traiettoria:
            self.get_logger().info(f"Stato iniziale: {np.rad2deg(self.q_current)}")
            self.get_logger().info("Avvio controllo cinematico!")
            self.timer = self.create_timer(self.dt, self.control_loop)"""


    # =========================================================
    # CINEMATICA
    # =========================================================

    def Ai_j(self, a, alpha, d, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),                d              ],
            [0,              0,                            0,                            1              ]
        ], dtype=float)

    def fk(self, q):
        T = np.eye(4)
        for i in range(6):
            T = T @ self.Ai_j(self.a[i], self.alpha[i], self.d[i], q[i])
        T_tool = np.eye(4)
        T_tool[2, 3] =  0.137#0.10
        #T_tool[0, 3] = -0.03
        return T @ T_tool

    def jacobian(self, q, eps=1e-6):
        J  = np.zeros((6, 6))
        T0 = self.fk(q)
        p0 = T0[0:3, 3]
        R0 = T0[0:3, 0:3]
        for i in range(6):
            dq    = q.copy()
            dq[i] += eps
            Ti = self.fk(dq)
            pi = Ti[0:3, 3]
            Ri = Ti[0:3, 0:3]
            J[0:3, i] = (pi - p0) / eps
            dR = Ri @ R0.T
            J[3:6, i] = np.array([
                dR[2, 1] - dR[1, 2],
                dR[0, 2] - dR[2, 0],
                dR[1, 0] - dR[0, 1]
            ]) / (2 * eps)
        return J

    def dls_inverse(self, J):
        lam = self.lambda_dls
        return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(6))

    def orientation_error(self, R_cur, R_des):
        Re = R_des @ R_cur.T
        return 0.5 * np.array([
            Re[2, 1] - Re[1, 2],
            Re[0, 2] - Re[2, 0],
            Re[1, 0] - Re[0, 1]
        ])


    # =========================================================
    # TRAIETTORIA — identica all'originale
    # =========================================================

    def genera_traiettoria_diversa(self):
        if self.q_current is not None:
            P_start = self.fk(self.q_current)[0:3, 3]
        else:
            P_start = np.array([-0.4869, -0.10915, 0.431859])

        """
        polsino_sx_raw = self.rete_to_base(np.array([[-369.19, 203.18, 515.0], [-286.19, 234.15, 515.0]]))
        self.polsino_sx_fitto = self.prendi_punti_intermedi(polsino_sx_raw[0], polsino_sx_raw[1])

        polsino_dx_raw = self.rete_to_base(np.array([[294.86, 229.2, 515.0], [377.87, 205.66, 515.0]]))
        self.polsino_dx_fitto = self.prendi_punti_intermedi(polsino_dx_raw[0], polsino_dx_raw[1])

        self.fondo_maglia = self.rete_to_base(np.array([
            [-151.15, 208.14, 515.0], [-117.7, 210.61, 515.0],
            [-85.48,  208.14, 515.0], [-52.03, 208.14, 515.0],
            [-19.82,  208.14, 515.0], [13.63,  206.9,  515.0],
            [45.84,   205.66, 515.0], [79.29,  205.66, 515.0],
            [111.5,   208.14, 515.0], [144.95, 208.14, 515.0]
        ]))

        self.colletto = self.rete_to_base(np.array([
            [-81.77,  -253.98, 515.0], [-60.71, -241.59, 515.0],
            [-39.64,  -204.42, 515.0], [-17.34, -193.27, 515.0],
            [3.72,    -192.03, 515.0], [24.78,  -190.79, 515.0],
            [45.84,   -194.51, 515.0], [68.14,  -205.66, 515.0],
            [89.2,    -242.83, 515.0], [110.26, -252.74, 515.0],
        ]))"""

        #CAMERA VERA

        """polsino_sx_raw = self.rete_to_base(np.array([[-299.44, 195.79, 787.0], [-177.87, 204.75, 787.0]]))
        self.polsino_sx_fitto = self.prendi_punti_intermedi(polsino_sx_raw[0], polsino_sx_raw[1])

        polsino_dx_raw = self.rete_to_base(np.array([[304.39, 176.91, 800.0], [411.33, 168.68, 798.0]]))
        self.polsino_dx_fitto = self.prendi_punti_intermedi(polsino_dx_raw[0], polsino_dx_raw[1])

        self.fondo_maglia = self.rete_to_base(np.array([
            [-102.67, 185.3, 770.0],
            [-66.37, 188.91, 785.0],
            [-28.19, 185.79, 788.0],
            [10.21, 181.25, 785.0],
            [48.57, 181.48, 786.0],
            [88.52, 180.89, 789.0],
            [126.69, 186.83, 787.0],
            [164.87, 189.15, 786.0],
            [203.47, 190.67, 787.0],
            [243.7, 194.7, 793.0],
        ]))

        self.colletto = self.rete_to_base(np.array([
            [-54.84, -358.09, 992.0],
            [-28.39, -340.68, 970.0],
            [-3.11, -295.66, 957.0],
            [21.69, -282.03, 953.0],
            [46.24, -280.55, 948.0],
            [69.73, -282.03, 953.0],
            [95.02, -283.51, 958.0],
            [119.94, -297.53, 958.0],
            [148.04, -340.66, 979.0],
            [177.06, -354.12, 999.0],
        ]))"""

        #polsino_sx_raw = self.rete_to_base_tf(np.array([[-360.43, 214.64, 740.0], [-249.69, 219.28, 756.0]]))
        polsino_sx_raw = self.rete_to_base_tf(np.array([[-409.07, 171.32, 921.0], [-284.72, 198.11, 909.0]]))
        self.polsino_sx_fitto = self.prendi_punti_intermedi(polsino_sx_raw[0], polsino_sx_raw[1])
        #self.polsino_sx_fitto[:, 1] -= 0.09
        self.polsino_sx_fitto = self.costante_z(self.polsino_sx_fitto, usa_iqr=False)

        #polsino_dx_raw = self.rete_to_base_tf(np.array([[238.63, 246.57, 817.5], [363.9, 250.31, 848.5]]))
        polsino_dx_raw = self.rete_to_base_tf(np.array([[332.86, 221.15, 889.0], [454.31, 217.7, 899.0]]))
        self.polsino_dx_fitto = self.prendi_punti_intermedi(polsino_dx_raw[0], polsino_dx_raw[1])
        #self.polsino_dx_fitto[:, 1] -= 0.09
        self.polsino_dx_fitto = self.costante_z(self.polsino_dx_fitto, usa_iqr=False)

        self.fondo_maglia = self.rete_to_base_tf(np.array([
            [-153.65, 209.32, 897.0],
            [-112.18, 205.49, 889.0],
            [7.19, 185.1, 894.5],
            [47.09, 185.29, 886.0],
            [164.52, 218.42, 878.0],
            [205.52, 222.83, 884.0],
        ]))
        #self.fondo_maglia[:, 1] -= 0.09
        self.fondo_maglia = self.costante_z(self.fondo_maglia, usa_iqr=False)

        """self.colletto = self.rete_to_base_tf(np.array([
            [-106.5, -323.96, 859.0],
            [-59.51, -271.94, 856.0],
            [-36.87, -247.61, 855.0],
            [-14.37, -246.22, 860.0],
            [9.78, -248.22, 862.0],
            [32.75, -251.23, 867.5],
            [56.17, -262.23, 875.5],
            [105.74, -320.02, 891.5],
        ]))"""
        self.colletto = self.rete_to_base_tf(np.array([
            [-67.15, -322.02, 1022.5],
            [-17.3, -275.85, 1002.0],
            [6.9, -251.76, 994.0],
            [30.83, -244.2, 990.0],
            [56.28, -245.93, 997.0],
            [80.66, -246.67, 1000.0],
            [104.97, -253.53, 1001.0],
            [155.88, -302.37, 1017.0],
        ]))
        #self.colletto[:, 1] -= 0.09
        self.colletto = self.costante_z(self.colletto, usa_iqr=False)

        """z_ref = np.mean(self.polsino_sx_fitto[:, 2])
        self.polsino_dx_fitto[:, 2] = z_ref
        self.fondo_maglia[:, 2]     = z_ref
        self.colletto[:, 2]         = z_ref"""
        z_ref = np.mean(self.fondo_maglia[:, 2])
        self.polsino_dx_fitto[:, 2] = z_ref
        self.polsino_sx_fitto[:, 2]     = z_ref
        self.colletto[:, 2]         = z_ref
                

        print("Polsino sx: ", self.polsino_sx_fitto)
        print("Polsino dx: ", self.polsino_dx_fitto)
        print("Fondo maglia: ", self.fondo_maglia)
        print("Colletto: ", self.colletto)


        trapezoids = self.build_trapezoids(P_start)

        all_x, all_y, all_z     = [], [], []
        all_vx, all_vy, all_vz  = [], [], []
        all_R                   = []

        if self.q_current is not None:
            R_current = self.fk(self.q_current)[0:3, 0:3]
        else:
            R_current = np.eye(3)

        R_init = {
            1: self.get_first_orientation(trapezoids[1]),
            3: self.get_first_orientation(trapezoids[3]),
            5: self.get_first_orientation(trapezoids[5]),
            7: self.get_first_orientation(trapezoids[7]),
        }

        last_R = R_current
        for i, trap in enumerate(trapezoids):
            is_transition = i in (0, 2, 4, 6)
            is_colletto = (i == len(trapezoids) - 1)  # ultimo trapezio = cucitura colletto

            """v_max_i = 0.02 if is_colletto else 0.04
            a_max_i = 0.0025 if is_colletto else 0.0045"""
            v_max_i = 0.03 if is_colletto else 0.06
            a_max_i = 0.002 if is_colletto else 0.005

            if i == 0 or i == 6:
                v_max_i = 0.2#0.06
                a_max_i = 0.06#0.006

            if is_transition:
                x, y, z, vx, vy, vz, Rot, t = self.genera_traiettoria_da_punti(
                    trap, interpolate_orientation=(last_R, R_init[i + 1]),
                    v_max=v_max_i, a_max=a_max_i
                )
            else:
                x, y, z, vx, vy, vz, Rot, t = self.genera_traiettoria_da_punti(
                    trap, v_max=v_max_i, a_max=a_max_i
                )
            last_R = Rot[-1]
            all_x.append(x);   all_y.append(y);   all_z.append(z)
            all_vx.append(vx); all_vy.append(vy); all_vz.append(vz)
            all_R.append(Rot)

        self.x_d  = np.concatenate(all_x)
        self.y_d  = np.concatenate(all_y)
        self.z_d  = np.concatenate(all_z)
        self.vx_d = np.concatenate(all_vx)
        self.vy_d = np.concatenate(all_vy)
        self.vz_d = np.concatenate(all_vz)
        
        self.R_d  = np.vstack(all_R)
        self.N    = len(self.x_d)
        self.finito_traiettoria = True

        import matplotlib.pyplot as plt
                
        v_norm = np.sqrt(self.vx_d**2 + self.vy_d**2 + self.vz_d**2)
        t = np.arange(self.N) * self.dt

        plt.plot(t, v_norm)
        plt.xlabel("Tempo simulazione [s]")
        plt.ylabel("Norma velocità [m/s]")
        plt.title("Norma velocità lungo la traiettoria concatenata")
        plt.grid(True)
        plt.show()

        self.visualizza_traiettoria()
        self.visualizza_orientazione()
        self.pubblica_come_joint_trajectory_action()
        #self.pubblica_traiettoria_desiderata()
        self.get_logger().info(f"Traiettoria: {self.N} punti a {1/self.dt:.0f} Hz")

    def smussa_z(self, P, window=3):
        P = P.copy()
        z = P[:, 2]
        kernel = np.ones(window) / window
        z_pad = np.pad(z, (window//2, window//2), mode='edge')
        P[:, 2] = np.convolve(z_pad, kernel, mode='valid')
        return P

    def smussa_z_grezza(self, punti_raw, window=3):
        """Smussa la colonna z (profondità) sui punti grezzi, prima della trasformazione."""
        punti_raw = punti_raw.astype(float).copy()
        z = punti_raw[:, 2]
        kernel = np.ones(window) / window
        z_pad = np.pad(z, (window // 2, window // 2), mode='edge')
        punti_raw[:, 2] = np.convolve(z_pad, kernel, mode='valid')
        return punti_raw

    def costante_z(self, P, metodo='mediana', usa_iqr=True, k_iqr=1.0):
        """Sostituisce la colonna z di tutti i punti con un unico valore costante.

        Args:
            P: array (N,3) di punti.
            metodo: 'mediana' o 'media', applicato ai punti rimasti dopo il filtro IQR.
            usa_iqr: se True, scarta prima gli outlier di z secondo il criterio IQR.
            k_iqr: moltiplicatore dell'IQR per definire i limiti (1.5 = standard,
                valori più bassi tipo 1.0 sono più aggressivi nel filtrare).
        """
        P = P.copy()
        z = P[:, 2]

        if usa_iqr and len(z) >= 4:
            q1 = np.percentile(z, 25)
            q3 = np.percentile(z, 75)
            iqr = q3 - q1
            lower = q1 - k_iqr * iqr
            upper = q3 + k_iqr * iqr
            z_validi = z[(z >= lower) & (z <= upper)]

            print(f"z originali:     {z}")
            print(f"Q1={q1:.5f}, Q3={q3:.5f}, IQR={iqr:.5f}")
            print(f"range valido:    [{lower:.5f}, {upper:.5f}]")
            print(f"z scartati:      {z[(z < lower) | (z > upper)]}")
            print(f"z validi (n={len(z_validi)}/{len(z)})")

            if z_validi.size == 0:  # fallback di sicurezza, non dovrebbe mai succedere
                z_validi = z
        else:
            z_validi = z

        if metodo == 'mediana':
            z_costante = np.median(z_validi)
        else:
            z_costante = np.mean(z_validi)

        print(f"z costante scelta: {z_costante:.5f}\n")

        P[:, 2] = z_costante
        return P

    def costante_z_iqr_strict(self, P, q_low=25, q_high=75):
        """Calcola la z costante come media del 50% centrale (IQR stretto),
        ma la applica a TUTTI i punti originali, senza scartarli."""
        P = P.copy()
        z = P[:, 2]

        if len(z) < 4:
            z_costante = np.median(z)
        else:
            q1 = np.percentile(z, q_low)
            q3 = np.percentile(z, q_high)
            mask = (z >= q1) & (z <= q3)
            z_validi = z[mask]
            z_costante = np.mean(z_validi) if z_validi.size > 0 else np.median(z)

        P[:, 2] = z_costante
        return P

    def correggi_solo_z(self, points_3d, q_low=25, q_high=75):
        if len(points_3d) == 0:
            return points_3d

        corrected_points = points_3d.copy()
        z_coords = points_3d[:, 2]

        q1_gialla = np.percentile(z_coords, q_low)
        q3_verde = np.percentile(z_coords, q_high)

        mask_z_valide = (z_coords >= q1_gialla) & (z_coords <= q3_verde)
        z_target_sicura = np.median(z_coords[mask_z_valide])

        mask_z_anomale = ~mask_z_valide
        corrected_points[mask_z_anomale, 2] = z_target_sicura

        return corrected_points

    def rete_to_base(self, punti_rete: np.ndarray) -> np.ndarray:
        # Step 1: rete → camera (mm → m, inversione asse Y)
        p_camera = np.column_stack([
            punti_rete[:, 0] / 1000.0,
            -punti_rete[:, 1] / 1000.0,
            punti_rete[:, 2] / 1000.0,
        ])

        # Step 2: camera → base Unity
        """ GIUSTA PER VERA HOMEPOSITION
        T = np.array([
            [ 0.0,  1.0,  0.0,  0.52370],
            [ 0.0,  0.0, -1.0,  0.56416],
            [-1.0,  0.0,  0.0,  0.10915],
            [ 0.0,  0.0,  0.0,  1.0    ]
        ])"""

        T = np.array([
            [ 0.0109,  -0.97763,  -0.21006,  -0.3306],
            [ -0.00939,  0.20996, -0.97766,  0.91691],
            [0.99990,  0.01263,  -0.00689,  -0.10477],
            [ 0.0,  0.0,  0.0,  1.0    ]            
        ])


        ones         = np.ones((len(p_camera), 1))
        P_h          = np.hstack([p_camera, ones])
        P_base_unity = (T @ P_h.T).T[:, :3]

        # Step 3: base Unity → base ROS
        points_ros = np.zeros_like(P_base_unity)
        points_ros[:, 0] = -P_base_unity[:, 2]  # x_ros = -z_unity
        points_ros[:, 1] =  P_base_unity[:, 0]  # y_ros =  x_unity
        points_ros[:, 2] =  P_base_unity[:, 1]  # z_ros =  y_unity

        return points_ros
    

    def rete_to_base_tf(self, punti_rete: np.ndarray) -> np.ndarray:
        """
        Converte coordinate rete neurale (frame camera) → frame base robot,
        usando la trasformazione TF corrente camera_link -> base_link.
        La camera è montata sul wrist e si muove col robot, quindi la
        trasformazione va letta ad ogni chiamata.
        """
        # Step 1: rete → camera (mm → m, inversione asse Y come prima)
        p_camera = np.column_stack([
            punti_rete[:, 0] / 1000.0,
            punti_rete[:, 1] / 1000.0,
            punti_rete[:, 2] / 1000.0,
        ])

        # Step 2: ottieni trasformazione camera_link -> base_link da TF
        try:
            t = self.tf_buffer.lookup_transform(
                'base',
                'camera_link',
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF camera_link->base_link non disponibile: {e}")
            return None

        # Costruisci matrice di trasformazione omogenea da TF
        trans = t.transform.translation
        rot   = t.transform.rotation

        T = np.eye(4)
        T[0:3, 0:3] = Rotation.from_quat(
            [rot.x, rot.y, rot.z, rot.w]
        ).as_matrix()
        T[0:3, 3] = [trans.x, trans.y, trans.z]

        # Step 3: applica trasformazione
        ones   = np.ones((len(p_camera), 1))
        P_h    = np.hstack([p_camera, ones])
        P_base = (T @ P_h.T).T[:, :3]

        return P_base


    # =========================================================
    # TUTTO IL RESTO — identico all'originale
    # =========================================================

    def prendi_punti_intermedi(self, start, end):
        start = np.array(start, dtype=float)
        end   = np.array(end,   dtype=float)
        return np.array([
            start,
            start + (end - start) * (1/3),
            start + (end - start) * (2/3),
            end
        ])

    def get_first_orientation(self, trap):
        return self.genera_orientamenti(trap[:2])[0]

    def build_trapezoids(self, current_pos):
        def transition_points(seg_a, seg_b, z_raise_m=0.1):
            end_pt   = seg_a[-1]
            start_pt = seg_b[0]
            dist_xy  = np.linalg.norm(start_pt[:2] - end_pt[:2])
            if dist_xy >= 0.1:
                p1_xy = end_pt[:2] + (start_pt[:2] - end_pt[:2]) * (1/3)
                p2_xy = end_pt[:2] + (start_pt[:2] - end_pt[:2]) * (2/3)
                z_up  = end_pt[2] + z_raise_m
                mid_points = [
                    np.array([p1_xy[0], p1_xy[1], z_up]),
                    np.array([p2_xy[0], p2_xy[1], z_up]),
                ]
            elif dist_xy >= 0.03:
                mid_xy = (end_pt[:2] + start_pt[:2]) / 2
                z_up   = end_pt[2] + z_raise_m
                mid_points = [np.array([mid_xy[0], mid_xy[1], z_up])]
            else:
                mid_points = []
            return np.array([end_pt, *mid_points, start_pt])

        trap_1 = np.array([current_pos, self.polsino_sx_fitto[0]])
        trap_2 = self.polsino_sx_fitto
        trap_3 = transition_points(self.polsino_sx_fitto, self.fondo_maglia)
        trap_4 = self.fondo_maglia
        trap_5 = transition_points(self.fondo_maglia,     self.polsino_dx_fitto)
        trap_6 = self.polsino_dx_fitto
        trap_7 = transition_points(self.polsino_dx_fitto, self.colletto)
        trap_8 = self.colletto
        return [trap_1, trap_2, trap_3, trap_4, trap_5, trap_6, trap_7, trap_8]

    def genera_traiettoria_da_punti(self, P, fixed_orientation=None, interpolate_orientation=None, v_max=0.03, a_max=0.0035):

        s = np.zeros(len(P))
        for i in range(1, len(P)):
            s[i] = s[i-1] + np.linalg.norm(P[i] - P[i-1])
        s /= s[-1]

        sx = CubicSpline(s, P[:, 0], bc_type='natural')
        sy = CubicSpline(s, P[:, 1], bc_type='natural')
        sz = CubicSpline(s, P[:, 2], bc_type='natural')

        L_spline = self.calcola_lunghezza_spline_semplice(sx, sy, sz)
        s_array, v_array = self.genera_profilo_trapezoidale_spazio(L_spline, v_max=v_max, a_max=a_max)
        t, s_tr = self.converti_spazio_in_tempo(s_array, v_array)

        s_tr_norm  = np.clip(s_tr / L_spline, 0.0, 1.0)
        s_arr_norm = s_array / L_spline

        v_interp   = interp1d(s_arr_norm, v_array, kind='linear',
                              bounds_error=False, fill_value=(v_array[0], v_array[-1]))
        v_at_pts   = v_interp(s_tr_norm)
        norm       = np.maximum(np.sqrt(sx.derivative()(s_tr_norm)**2 +
                                        sy.derivative()(s_tr_norm)**2 +
                                        sz.derivative()(s_tr_norm)**2), 1e-6)

        x_d  = sx(s_tr_norm)
        y_d  = sy(s_tr_norm)
        z_d  = sz(s_tr_norm)
        vx_d = (sx.derivative()(s_tr_norm) / norm) * v_at_pts
        vy_d = (sy.derivative()(s_tr_norm) / norm) * v_at_pts
        vz_d = (sz.derivative()(s_tr_norm) / norm) * v_at_pts

        P_traj = np.column_stack((x_d, y_d, z_d))

        if interpolate_orientation is not None:
            R_start, R_end = interpolate_orientation
            slerp = Slerp([0.0, 1.0], Rotation.concatenate([
                Rotation.from_matrix(R_start),
                Rotation.from_matrix(R_end)
            ]))
            R_d = [slerp(a).as_matrix() for a in s_tr_norm]
        elif fixed_orientation is not None:
            R_d = [fixed_orientation] * len(x_d)
        else:
            R_d = self.genera_orientamenti(P_traj)

        return x_d, y_d, z_d, vx_d, vy_d, vz_d, R_d, t

    def genera_orientamenti_prima(self, P):
        R_list = []
        for k in range(len(P) - 1):
            tang = P[k+1] - P[k]
            x_ee = tang / np.linalg.norm(tang) if np.linalg.norm(tang) > 1e-6 else (
                R_list[-1][:, 0] if R_list else np.array([1., 0., 0.])
            )
            z_ee = np.array([0., 0., -1.])
            y_ee = np.cross(z_ee, x_ee)
            if np.linalg.norm(y_ee) > 1e-6:
                y_ee /= np.linalg.norm(y_ee)
            else:
                y_ee = np.array([-x_ee[1], x_ee[0], 0.])
                y_ee /= np.linalg.norm(y_ee)
            z_ee = np.cross(x_ee, y_ee)
            z_ee /= np.linalg.norm(z_ee)
            R_list.append(np.column_stack((x_ee, y_ee, z_ee)))
        R_list.append(R_list[-1])

        quats = np.array([Rotation.from_matrix(R).as_quat() for R in R_list])
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i-1]) < 0:
                quats[i] = -quats[i]
        alpha = 0.2
        quats_f = quats.copy()
        for i in range(1, len(quats)):
            quats_f[i] = (1 - alpha) * quats_f[i-1] + alpha * quats[i]
            quats_f[i] /= np.linalg.norm(quats_f[i])
        return [Rotation.from_quat(q).as_matrix() for q in quats_f]

    def genera_orientamenti(self, P):
        n = len(P)
        tang = np.zeros((n, 3))
        for k in range(n - 1):
            d = P[k+1] - P[k]
            norm = np.linalg.norm(d)
            tang[k] = d / norm if norm > 1e-6 else (tang[k-1] if k > 0 else np.array([1., 0., 0.]))
        tang[-1] = tang[-2]

        z_ref = np.array([0., 0., -1.])
        R_list = []
        y_prev = None

        # soglia "morbida": sotto questo valore iniziamo a miscelare con il frame precedente
        blend_start = 0.25   # inizia a miscelare quando |sin(angolo)| < 0.25
        blend_end   = 0.03   # sotto questo valore uso quasi solo il frame precedente

        for k in range(n):
            x_ee = tang[k]

            # candidato "standard": y da cross con riferimento verticale fisso
            y_std = np.cross(z_ref, x_ee)
            norm_std = np.linalg.norm(y_std)

            if norm_std > 1e-9:
                y_std = y_std / norm_std

            if y_prev is None:
                # primo campione: nessun riferimento precedente, uso lo standard
                # (con fallback solo se davvero degenere)
                if norm_std > 1e-6:
                    y_ee = y_std
                else:
                    y_ee = np.array([-x_ee[1], x_ee[0], 0.])
                    y_ee /= np.linalg.norm(y_ee)
            else:
                # candidato "continuo": proietto y_prev nel piano perpendicolare a x_ee
                y_cont = y_prev - np.dot(y_prev, x_ee) * x_ee
                n_cont = np.linalg.norm(y_cont)
                y_cont = y_cont / n_cont if n_cont > 1e-9 else y_prev

                if norm_std >= blend_start:
                    y_ee = y_std
                elif norm_std <= blend_end:
                    y_ee = y_cont
                else:
                    # interpolazione morbida tra i due (poi rinormalizzo)
                    w = (norm_std - blend_end) / (blend_start - blend_end)  # 0..1
                    y_ee = w * y_std + (1 - w) * y_cont
                    y_ee /= np.linalg.norm(y_ee)

            z_ee = np.cross(x_ee, y_ee)
            z_ee /= np.linalg.norm(z_ee)
            # riortogonalizzo y per sicurezza numerica
            y_ee = np.cross(z_ee, x_ee)
            y_ee /= np.linalg.norm(y_ee)

            R_list.append(np.column_stack((x_ee, y_ee, z_ee)))
            y_prev = y_ee

        # filtro passa-basso sui quaternioni, come avevi già
        quats = np.array([Rotation.from_matrix(R).as_quat() for R in R_list])
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i-1]) < 0:
                quats[i] = -quats[i]
        alpha = 0.2
        quats_f = quats.copy()
        for i in range(1, len(quats)):
            quats_f[i] = (1 - alpha) * quats_f[i-1] + alpha * quats[i]
            quats_f[i] /= np.linalg.norm(quats_f[i])
        return [Rotation.from_quat(q).as_matrix() for q in quats_f]

    def calcola_lunghezza_spline_semplice(self, sx, sy, sz):
        s = np.linspace(0, 1, 1000)
        return np.sum(np.sqrt(np.diff(sx(s))**2 + np.diff(sy(s))**2 + np.diff(sz(s))**2))

    def genera_profilo_trapezoidale_spazio(self, L_total, v_max, a_max):
        s_accel = v_max**2 / (2 * a_max)
        if 2 * s_accel > L_total:
            a_max   = v_max**2 / L_total
            s_accel = L_total / 2
            s_coast = 0
        else:
            s_coast = L_total - 2 * s_accel
        v_peak  = v_max
        n       = 1000
        s_array = np.linspace(0, L_total, n)
        v_array = np.zeros(n)
        for i, s in enumerate(s_array):
            if s < s_accel:
                v_array[i] = np.sqrt(2 * a_max * s)
            elif s < s_accel + s_coast:
                v_array[i] = v_peak
            else:
                v_array[i] = np.sqrt(max(2 * a_max * (L_total - s), 0))
        return s_array, v_array

    def converti_spazio_in_tempo(self, s_array, v_array):
        t_cum = [0.0]
        for i in range(1, len(s_array)):
            ds    = s_array[i] - s_array[i-1]
            v_avg = (v_array[i] + v_array[i-1]) / 2
            t_cum.append(t_cum[-1] + (ds / v_avg if v_avg > 1e-6 else 0.0))
        t_cum   = np.array(t_cum)
        T_total = t_cum[-1]
        self.get_logger().info(f"Durata totale: {T_total:.3f} s")
        s_of_t  = interp1d(t_cum, s_array, kind='linear',
                           bounds_error=False, fill_value=(s_array[0], s_array[-1]))
        n       = int(np.floor(T_total / self.dt)) + 1
        t_unif  = np.linspace(0, T_total, n)
        s_time  = np.clip(s_of_t(t_unif), s_array[0], s_array[-1])
        self.get_logger().info(f"Punti: {len(t_unif)}")
        return t_unif, s_time

    def visualizza_traiettoria(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        cm = 100.0
        x, y, z = self.x_d * cm, self.y_d * cm, self.z_d * cm

        segs = {
            'Polsino SX':   (self.polsino_sx_fitto, 'tab:green',  'o'),
            'Fondo maglia': (self.fondo_maglia,      'tab:red',    's'),
            'Polsino DX':   (self.polsino_dx_fitto, 'tab:blue',   '^'),
            'Colletto':     (self.colletto,          'tab:purple', '*'),
        }

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Traiettoria cartesiana', fontsize=14, fontweight='bold')

        # ---------- 1) Vista 3D ----------
        ax3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax3d.plot(x, y, z, '-', color='dimgray', linewidth=1.2, alpha=0.7, label='Traiettoria')
        for label, (pts, color, marker) in segs.items():
            ax3d.scatter(pts[:, 0]*cm, pts[:, 1]*cm, pts[:, 2]*cm,
                        color=color, marker=marker, s=50, label=label, depthshade=False)
        ax3d.scatter(x[0], y[0], z[0], c='k', marker='X', s=90, label='Start')
        ax3d.set_xlabel('X [cm]'); ax3d.set_ylabel('Y [cm]'); ax3d.set_zlabel('Z [cm]')
        ax3d.set_title('Vista 3D')
        try:
            ax3d.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
        except Exception:
            pass

        # ---------- 2) Vista dall'alto (XY) ----------
        axxy = fig.add_subplot(2, 2, 2)
        axxy.plot(x, y, '-', color='dimgray', linewidth=1.2, alpha=0.7)
        for label, (pts, color, marker) in segs.items():
            axxy.scatter(pts[:, 0]*cm, pts[:, 1]*cm, color=color, marker=marker, s=50, label=label)
        axxy.scatter(x[0], y[0], c='k', marker='X', s=90, label='Start')
        axxy.set_xlabel('X [cm]'); axxy.set_ylabel('Y [cm]')
        axxy.set_title('Vista dall\'alto (piano XY)')
        axxy.set_aspect('equal', adjustable='box')
        axxy.grid(True, alpha=0.3)

        # ---------- 3) Vista frontale (XZ) ----------
        axxz = fig.add_subplot(2, 2, 3)
        axxz.plot(x, z, '-', color='dimgray', linewidth=1.2, alpha=0.7)
        for label, (pts, color, marker) in segs.items():
            axxz.scatter(pts[:, 0]*cm, pts[:, 2]*cm, color=color, marker=marker, s=50, label=label)
        axxz.scatter(x[0], z[0], c='k', marker='X', s=90, label='Start')
        axxz.set_xlabel('X [cm]'); axxz.set_ylabel('Z [cm]')
        axxz.set_title('Vista frontale (piano XZ)')
        axxz.grid(True, alpha=0.3)

        # ---------- 4) Vista laterale (YZ) ----------
        axyz = fig.add_subplot(2, 2, 4)
        axyz.plot(y, z, '-', color='dimgray', linewidth=1.2, alpha=0.7)
        for label, (pts, color, marker) in segs.items():
            axyz.scatter(pts[:, 1]*cm, pts[:, 2]*cm, color=color, marker=marker, s=50, label=label)
        axyz.scatter(y[0], z[0], c='k', marker='X', s=90, label='Start')
        axyz.set_xlabel('Y [cm]'); axyz.set_ylabel('Z [cm]')
        axyz.set_title('Vista laterale (piano YZ)')
        axyz.grid(True, alpha=0.3)

        handles, labels = axxy.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.show()
    

    def visualizza_orientazione(self, step=20, scale=0.03):
        """
        Visualizza l'orientazione lungo la traiettoria:
        - subplot 3D con le terne (x=rosso, y=verde, z=blu) ogni `step` campioni
        - subplot 2D con gli angoli di Eulero (roll, pitch, yaw) per ogni campione
        - subplot 2D con le componenti del quaternione (utile per scovare "flip")
        """
        import matplotlib.pyplot as plt
        from scipy.spatial.transform import Rotation as Rsc

        if self.R_d is None or self.N == 0:
            print("Nessuna traiettoria disponibile: genera prima la traiettoria.")
            return

        fig = plt.figure(figsize=(16, 6))
            
        # --- 1) Terne lungo la traiettoria in 3D ---
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot(self.x_d, self.y_d, self.z_d, 'k-', linewidth=1, label='traiettoria')

        for i in range(0, self.N, step):
            origin = np.array([self.x_d[i], self.y_d[i], self.z_d[i]])
            Rot = self.R_d[i]
            x_axis, y_axis, z_axis = Rot[:, 0], Rot[:, 1], Rot[:, 2]
            ax1.quiver(*origin, *x_axis, length=scale, color='r', normalize=True)
            ax1.quiver(*origin, *y_axis, length=scale, color='g', normalize=True)
            ax1.quiver(*origin, *z_axis, length=scale, color='b', normalize=True)

        ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
        ax1.set_title('Terne orientazione (rosso=x, verde=y, blu=z)')
        ax1.legend()


        # --- 2) Angoli di Eulero vs indice campione ---
        eulers = Rsc.from_matrix(self.R_d).as_euler('xyz', degrees=True)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(eulers[:, 0], label='roll (x)')
        ax2.plot(eulers[:, 1], label='pitch (y)')
        ax2.plot(eulers[:, 2], label='yaw (z)')
        ax2.set_xlabel('campione'); ax2.set_ylabel('gradi')
        ax2.set_title('Angoli di Eulero lungo la traiettoria')
        ax2.legend()
        ax2.grid(True)

        # --- 3) Quaternioni vs indice campione (per scovare flip di segno) ---
        quats = Rsc.from_matrix(self.R_d).as_quat()  # [x, y, z, w]
        ax3 = fig.add_subplot(1, 3, 3)
        labels = ['qx', 'qy', 'qz', 'qw']
        for k in range(4):
            ax3.plot(quats[:, k], label=labels[k])
        ax3.set_xlabel('campione'); ax3.set_ylabel('valore')
        ax3.set_title('Componenti quaternione')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

    def pubblica_traiettoria_desiderata(self):
        pa = PoseArray()
        pa.header.frame_id = "base_link"
        pa.header.stamp    = self.get_clock().now().to_msg()
        for i in range(self.N):
            p = Pose()
            p.position.x = self.x_d[i]
            p.position.y = self.y_d[i]
            p.position.z = self.z_d[i]
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.pub_trajectory_desired.publish(pa)
        self.get_logger().info(f"Traiettoria pubblicata: {self.N} punti")


    # =========================================================
    # CONTROL LOOP
    # =========================================================

    def control_loop(self):
        if self.q_current is None:
            self.get_logger().warn("Stato robot non ancora ricevuto...")
            return

        if self.idx >= self.N:
            self.get_logger().info("Traiettoria completata!")
            self.timer.cancel()
            return

        xd = np.array([self.x_d[self.idx], self.y_d[self.idx], self.z_d[self.idx]])
        vd = np.array([self.vx_d[self.idx], self.vy_d[self.idx], self.vz_d[self.idx]])
        Rd = self.R_d[self.idx]

        q     = self.q_current.copy()
        T_cur = self.fk(q)
        x_cur = T_cur[0:3, 3]
        R_cur = T_cur[0:3, 0:3]

        ep    = xd - x_cur
        eo    = self.orientation_error(R_cur, Rd)
        e     = np.hstack((ep, eo))
        v_ref = np.hstack((vd, np.zeros(3))) + self.K @ e

        J     = self.jacobian(q)
        J_inv = self.dls_inverse(J)
        dq    = np.clip(J_inv @ v_ref, -self.dq_max, self.dq_max)

        """# Pubblica come JointTrajectory
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = self.joint_names

        point = JointTrajectoryPoint()
        point.positions   = list(q + dq * self.dt)
        point.velocities  = list(dq)
        point.time_from_start = Duration(sec=0, nanosec=int(self.dt * 1e9))

        traj.points = [point]
        self.pub_joint.publish(traj)"""

        # Nel control_loop, sostituisci tutto il blocco JointTrajectory con:
        msg = Float64MultiArray()
        msg.data = list(q + dq * self.dt)
        self.pub_joint.publish(msg)

        if self.idx % 20 == 0:
            self.get_logger().info(
                f"t={self.idx*self.dt:.2f}s | "
                f"|ep|={np.linalg.norm(ep)*1000:.2f}mm | "
                f"|eo|={np.linalg.norm(eo)*180/np.pi:.2f}°"
            )

        self.idx += 1

    def calcola_traiettoria_giunti_offline(self):
        """
        Integra la cinematica differenziale (stessa logica di control_loop)
        su tutta la traiettoria cartesiana pre-calcolata, restituendo
        posizione e velocità articolare per ogni campione.
        """
        q_traj  = np.zeros((self.N, 6))
        dq_traj = np.zeros((self.N, 6))

        q = self.q_current.copy()

        for k in range(self.N):
            xd = np.array([self.x_d[k], self.y_d[k], self.z_d[k]])
            vd = np.array([self.vx_d[k], self.vy_d[k], self.vz_d[k]])
            Rd = self.R_d[k]

            T_cur = self.fk(q)
            x_cur = T_cur[0:3, 3]
            R_cur = T_cur[0:3, 0:3]

            ep = xd - x_cur
            eo = self.orientation_error(R_cur, Rd)
            e  = np.hstack((ep, eo))
            v_ref = np.hstack((vd, np.zeros(3))) + self.K @ e

            J     = self.jacobian(q)
            J_inv = self.dls_inverse(J)
            dq    = np.clip(J_inv @ v_ref, -self.dq_max, self.dq_max)

            q_traj[k]  = q
            dq_traj[k] = dq

            # integrazione in avanti: il q "attuale" per il prossimo campione
            q = q + dq * self.dt

        return q_traj, dq_traj


    def pubblica_come_joint_trajectory_action(self):
        if self.q_current is None:
            self.get_logger().error("Stato robot non disponibile!")
            return

        self.get_logger().info("Calcolo traiettoria articolare completa (offline)...")
        q_traj, dq_traj = self.calcola_traiettoria_giunti_offline()

        self.get_logger().info("In attesa dell'action server...")
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server non disponibile!")
            return

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        points = []
        for k in range(self.N):
            point = JointTrajectoryPoint()
            point.positions  = q_traj[k].tolist()
            point.velocities = dq_traj[k].tolist()
            t_sec = (k + 1) * self.dt
            point.time_from_start = Duration(
                sec=int(t_sec),
                nanosec=int(round((t_sec - int(t_sec)) * 1e9))
            )
            points.append(point)

        traj.points = points

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj
        # tolleranze generose, giusto per il test - non vogliamo che fallisca per questo
        goal_msg.goal_time_tolerance = Duration(sec=5, nanosec=0)

        self.get_logger().info(f"Invio goal: {self.N} punti, durata {self.N * self.dt:.2f}s")
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(">>> GOAL RIFIUTATO dal controller! <<<")
            return

        self.get_logger().info("Goal ACCETTATO, esecuzione in corso...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)


    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f">>> Esecuzione terminata. error_code={result.error_code}, error_string='{result.error_string}' <<<")


def main(args=None):
    rclpy.init(args=args)
    node = KinematicController()
    rclpy.spin(node)
    rclpy.shutdown()