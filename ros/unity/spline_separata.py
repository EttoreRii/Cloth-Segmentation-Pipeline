#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray, Pose


class KinematicController(Node):

    def __init__(self):
        super().__init__("kinematic_controller")

        # Parametri DH UR5
        self.a     = [0,       -0.425,   -0.39225, 0,       0,       0      ]
        self.alpha = [np.pi/2,  0,        0,        np.pi/2, -np.pi/2, 0     ]
        self.d     = [0.089159, 0,        0,        0.10915, 0.09465, 0.0823 ]

        # Limiti giunti UR5
        self.q_min = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.q_max = np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])

        # Limiti velocità UR5 [rad/s]
        self.dq_max = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28])

        # Guadagni del controllore (tuning!)
        self.Kp = 14.5#8.8#8.7=BUONO   # 7.7 Guadagno proporzionale posizione
        self.Ko = 5.0#3.4#4.0   # Guadagno proporzionale orientamento
        self.K  = np.diag([self.Kp, self.Kp, self.Kp,
                           self.Ko, self.Ko, self.Ko])

        # Damping per DLS (Damped Least Squares)
        self.lambda_dls = 0.03

        # Configurazione iniziale (home)
        self.q = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])

        # Publishers / Subscribers
        self.pub_joint = self.create_publisher(JointState, '/joint_targets', 10)
        self.pub_pose  = self.create_publisher(Pose, '/ee_pose_debug', 10)
        self.pub_trajectory_desired = self.create_publisher(PoseArray, '/trajectory_desired', 10)

        # Timer e indice traiettoria
        self.timer = None
        self.idx   = 0
        self.dt    = 0.05

        self.finito_traiettoria = False


        # Subscriber allo stato del robot
        self.sub_joint = self.create_subscription(
            JointState,
            '/joint_states',  # Topic pubblicato da Unity
            self.joint_state_callback,
            10
        )

        # Stato attuale ricevuto dal robot/simulazione
        self.q_current = None  # None finché non arriva il primo messaggio
        self.q_current_lock = False  # Flag per evitare race conditions
        


        # Avvia controllo
        self.get_logger().info("Avvio controllo cinematico...")
        #ASPETTA
        # self.timer = self.create_timer(self.dt, self.control_loop)

    
    def joint_state_callback(self, msg: JointState):
        """
        Riceve la configurazione attuale dei giunti dal robot/Unity.
        Riordina i giunti nel giusto ordine (Unity potrebbe mandarli in ordine diverso!)
        """
        joint_order = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # Riordina i giunti nell'ordine corretto
        q = np.zeros(6)
        for i, name in enumerate(joint_order):
            if name in msg.name:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]
            else:
                self.get_logger().warn(f"Giunto {name} non trovato in /joint_states!")
                return

        self.q_current = q

        # Genera traiettoria solo la prima volta
        if not self.finito_traiettoria:
            self.get_logger().info("Generazione traiettoria da posizione corrente...")
            self.genera_traiettoria_diversa()
            #self.plot_3d_trajectory_with_waypoints(self.P)

        # Avvia il timer di controllo solo quando riceviamo
        # la prima configurazione valida
        if self.timer is None and self.q_current is not None and self.finito_traiettoria is True:
            self.get_logger().info(f"Stato iniziale ricevuto: {np.rad2deg(self.q_current)}")
            self.get_logger().info("Avvio controllo cinematico!")
            self.timer = self.create_timer(self.dt, self.control_loop)


    # =========================================================
    # CINEMATICA
    # =========================================================

    def Ai_j(self, a, alpha, d, theta):
        """Matrice DH numerica"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),                d              ],
            [0,              0,                            0,                            1              ]
        ], dtype=float)

    def fk(self, q):
        """Cinematica diretta: restituisce T_06 (4x4)"""
        T = np.eye(4)
        for i in range(6):
            T = T @ self.Ai_j(self.a[i], self.alpha[i], self.d[i], q[i])
        

        # ── Tool transform ───────────────────────────────────────────
        T_tool = np.eye(4)
        T_tool[2, 3] = 0.10  # 10cm lungo Z del flange
        T_tool[0, 3] = -0.03 #forse metterei -0.028
        
        T = T @ T_tool
        # ────────────────────────────────────────────────────────────
        return T

    def jacobian(self, q, eps=1e-6):
        """
        Jacobiano geometrico numerico (6x6)
        Righe 0:3 → velocità lineare
        Righe 3:6 → velocità angolare
        """
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

            # Jacobiano lineare
            J[0:3, i] = (pi - p0) / eps

            # Jacobiano angolare (approssimazione skew-symmetric)
            dR = Ri @ R0.T
            J[3:6, i] = np.array([
                dR[2, 1] - dR[1, 2],
                dR[0, 2] - dR[2, 0],
                dR[1, 0] - dR[0, 1]
            ]) / (2 * eps)

        return J

    def dls_inverse(self, J):
        """
        Pseudo-inversa con Damped Least Squares
        Evita singolarità: J^T (J J^T + λ²I)^{-1}
        """
        lam = self.lambda_dls
        return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(6))

    # =========================================================
    # ERRORE ORIENTAMENTO
    # =========================================================

    def orientation_error(self, R_cur, R_des):
        """
        Errore di orientamento basato su asse-angolo
        e_o = 0.5 * (R_des × R_cur^T) estratto come vettore
        """
        Re = R_des @ R_cur.T
        return 0.5 * np.array([
            Re[2, 1] - Re[1, 2],
            Re[0, 2] - Re[2, 0],
            Re[1, 0] - Re[0, 1]
        ])

    # =========================================================
    # TRAIETTORIA CARTESIANA
    # =========================================================

    def genera_traiettoria_diversa(self):

        # ── Setup punti ────────────────────────────────────────────────
        if self.q_current is not None:
            P_start = self.fk(self.q_current)[0:3, 3]
        else:
            P_start = np.array([-0.4869, -0.10915, 0.431859])


        """p = self.rete_to_base(np.array([[109.15, 23.70, 464.16]]))
        print("Python:", p)
        print("Atteso ROS (converti inBase con x=-z, y=x, z=y):", 
            np.array([-p[:, 2], p[:, 0], p[:, 1]]))"""
        

        polsino_sx_raw = self.rete_to_base(np.array([[-369.19, 203.18, 515.0], [-286.19, 234.15, 515.0]]))
        self.polsino_sx_fitto = self.prendi_punti_intermedi(polsino_sx_raw[0], polsino_sx_raw[1])

        polsino_dx_raw = self.rete_to_base(np.array([[294.86, 229.2, 515.0], [377.87, 205.66, 515.0]]))
        self.polsino_dx_fitto = self.prendi_punti_intermedi(polsino_dx_raw[0], polsino_dx_raw[1])

        self.fondo_maglia = self.rete_to_base(np.array([
            [-151.15, 208.14, 515.0],
            [-117.7, 210.61, 515.0],
            [-85.48, 208.14, 515.0],
            [-52.03, 208.14, 515.0],
            [-19.82, 208.14, 515.0],
            [13.63, 206.9, 515.0],
            [45.84, 205.66, 515.0],
            [79.29, 205.66, 515.0],
            [111.5, 208.14, 515.0],
            [144.95, 208.14, 515.0]
        ]))

        self.colletto = self.rete_to_base(np.array([
             [-81.77, -253.98, 515.0],
            [-60.71, -241.59, 515.0],
            [-39.64, -204.42, 515.0],
            [-17.34, -193.27, 515.0],
            [3.72, -192.03, 515.0],
            [24.78, -190.79, 515.0],
            [45.84, -194.51, 515.0],
            [68.14, -205.66, 515.0],
            [89.2, -242.83, 515.0],
            [110.26, -252.74, 515.0],
        ]))
        
        
        # ── 7 trapezi ──────────────────────────────────────────────────
        trapezoids = self.build_trapezoids(P_start)  # già in metri


        # ── Genera e concatena ─────────────────────────────────────────
        all_x, all_y, all_z     = [], [], []
        all_vx, all_vy, all_vz  = [], [], []
        all_R                   = []
        t_offset = 0.0

        # Orientamento corrente del robot
        if self.q_current is not None:
            T_current = self.fk(self.q_current)
            R_current = T_current[0:3, 0:3]
        else:
            R_current = np.eye(3)  # fallback

        
        R_init = {
            # i trapzei vanno da 0 a 7, ne sono 8, i pari sono i raccordi
            1: self.get_first_orientation(trapezoids[1]),  # inizio trap_2 (polsino_sx)
            3: self.get_first_orientation(trapezoids[3]),  # inizio trap_4 (fondo_maglia)
            5: self.get_first_orientation(trapezoids[5]),  # inizio trap_6 (polsino_dx)
            7: self.get_first_orientation(trapezoids[7]),  # inizio trap_8 (colletto)
        }

        last_R = R_current

        
        for i, trap in enumerate(trapezoids):
            is_transition = i in (0, 2, 4, 6)

            if is_transition:
                # SLERP per tutti i trapezi di transizione
                x, y, z, vx, vy, vz, R, t = self.genera_traiettoria_da_punti(
                    trap,
                    interpolate_orientation=(last_R, R_init[i + 1])
                )
            else:
                x, y, z, vx, vy, vz, R, t = self.genera_traiettoria_da_punti(trap)

            last_R = R[-1]

            all_x.append(x);   all_y.append(y);   all_z.append(z)
            all_vx.append(vx); all_vy.append(vy); all_vz.append(vz)
            all_R.append(R)
            t_offset += t[-1]

        self.x_d  = np.concatenate(all_x)
        self.y_d  = np.concatenate(all_y)
        self.z_d  = np.concatenate(all_z)
        self.vx_d = np.concatenate(all_vx)
        self.vy_d = np.concatenate(all_vy)
        self.vz_d = np.concatenate(all_vz)
        self.R_d  = np.vstack(all_R)
        self.N    = len(self.x_d)

        self.finito_traiettoria = True

        self.pubblica_traiettoria_desiderata()
        self.get_logger().info(f"Traiettoria completa: {self.N} punti a {1/self.dt:.0f} Hz")

    

    def genera_traiettoria_da_punti(self, P: np.ndarray, fixed_orientation=None, interpolate_orientation=None):
        """
        Riutilizzabile per ogni trapezio.
        Prende punti già in metri, restituisce x_d, y_d, z_d, vx_d, vy_d, vz_d, R_d
        """
        

        # Ascissa curvilinea normalizzata
        s = np.zeros(len(P))
        for i in range(1, len(P)):
            s[i] = s[i-1] + np.linalg.norm(P[i] - P[i-1])
        s /= s[-1]

        sx = CubicSpline(s, P[:, 0], bc_type='natural')
        sy = CubicSpline(s, P[:, 1], bc_type='natural')
        sz = CubicSpline(s, P[:, 2], bc_type='natural')

        L_spline = self.calcola_lunghezza_spline_semplice(sx, sy, sz)

        v_max = 0.03
        a_max = 0.0035

        s_array, v_array = self.genera_profilo_trapezoidale_spazio(L_spline, v_max, a_max)
        t, s_tr          = self.converti_spazio_in_tempo(s_array, v_array)

        s_tr_norm   = np.clip(s_tr / L_spline, 0.0, 1.0)
        s_arr_norm  = s_array / L_spline

        v_interp    = interp1d(s_arr_norm, v_array, kind='linear',
                            bounds_error=False, fill_value=(v_array[0], v_array[-1]))
        v_at_points = v_interp(s_tr_norm)

        norm = np.sqrt(sx.derivative()(s_tr_norm)**2 +
                    sy.derivative()(s_tr_norm)**2 +
                    sz.derivative()(s_tr_norm)**2)
        
        norm = np.maximum(norm, 1e-6)  # evita divisione per zero

        x_d  = sx(s_tr_norm)
        y_d  = sy(s_tr_norm)
        z_d  = sz(s_tr_norm)
        vx_d = (sx.derivative()(s_tr_norm) / norm) * v_at_points
        vy_d = (sy.derivative()(s_tr_norm) / norm) * v_at_points
        vz_d = (sz.derivative()(s_tr_norm) / norm) * v_at_points

        P_traj = np.column_stack((x_d, y_d, z_d))

        # ── Orientamento ────────────────────────────────────────────────

        if interpolate_orientation is not None:
            # SLERP tra R_start e R_end lungo il trapezio
            R_start, R_end = interpolate_orientation
            r_start = Rotation.from_matrix(R_start)
            r_end   = Rotation.from_matrix(R_end)
            slerp   = Slerp([0.0, 1.0], Rotation.concatenate([r_start, r_end]))
            R_d = [slerp(alpha).as_matrix() for alpha in s_tr_norm]

        elif fixed_orientation is not None:
            R_d = [fixed_orientation] * len(x_d)

        else:
            R_d = self.genera_orientamenti(P_traj)

        return x_d, y_d, z_d, vx_d, vy_d, vz_d, R_d, t



    def prendi_punti_intermedi(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """
        Data una retta definita da start e end, restituisce 4 punti:
        - start (0%)
        - primo intermedio (1/3)
        - secondo intermedio (2/3)
        - end (100%)

        Args:
            start: array [x, y, z] punto iniziale manica
            end:   array [x, y, z] punto finale manica

        Returns:
            np.ndarray (4, 3)
        """
        start = np.array(start, dtype=float)
        end   = np.array(end,   dtype=float)

        p0 = start
        p1 = start + (end - start) * (1/3)
        p2 = start + (end - start) * (2/3)
        p3 = end

        return np.array([p0, p1, p2, p3])
    

    def get_first_orientation(self, trap: np.ndarray):
        """Calcola solo l'orientamento del primo punto di un trapezio."""
        P_traj_dummy = trap[:2]  # bastano 2 punti per la prima tangente
        R_list = self.genera_orientamenti(P_traj_dummy)
        return R_list[0]


    def build_trapezoids(self, current_pos: np.ndarray) -> list[np.ndarray]:
        """
        Costruisce 7 trapezi separati a partire dalla posizione corrente.

        Trapezi oggetto  (1,3,5,7): punti dell'elemento
        Trapezi transiz. (2,4,6):   punti di rialzo tra elementi

        Returns:
            lista di 7 array (N,3), uno per trapezio
        """
        

        # Calcola i punti di transizione tra ogni coppia di segmenti
        def transition_points(seg_a, seg_b, z_raise_m=0.1):
            end_pt   = seg_a[-1]
            start_pt = seg_b[0]
            dist_xy  = np.linalg.norm(start_pt[:2] - end_pt[:2])
            print(dist_xy)

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
                mid_points = [
                    np.array([mid_xy[0], mid_xy[1], z_up]),
                ]
            else:
                mid_points = []

            print("MID POINTS: ", mid_points)

            # il trapezio di transizione include: fine A + rialzi + inizio B
            return np.array([end_pt, *mid_points, start_pt])


        # 8 trapezi
        trap_1 = np.array([current_pos, self.polsino_sx_fitto[0]])
        trap_2 = self.polsino_sx_fitto
        trap_3 = transition_points(self.polsino_sx_fitto, self.fondo_maglia)
        trap_4 = self.fondo_maglia
        trap_5 = transition_points(self.fondo_maglia,     self.polsino_dx_fitto)
        trap_6 = self.polsino_dx_fitto
        trap_7 = transition_points(self.polsino_dx_fitto, self.colletto)
        trap_8 = self.colletto

        return [trap_1, trap_2, trap_3, trap_4, trap_5, trap_6, trap_7, trap_8]

    
    def rete_to_base(self, punti_rete: np.ndarray) -> np.ndarray:
        """
        Converte output rete neurale in coordinate base robot.
        """
        
        # Step 1: rete → camera Unity
        p_camera_unity = np.column_stack([
            punti_rete[:, 0] / 1000.0,
            -punti_rete[:, 1] / 1000.0,
            punti_rete[:, 2] / 1000.0,
        ])

        # Step 2: camera Unity → base Unity (euler xyz confermato)
        """R_unity = Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
        T_base_camera_unity = np.eye(4)
        T_base_camera_unity[0:3, 0:3] = R_unity
        T_base_camera_unity[0:3, 3]   = np.array([0.52, 0.56, -0.11])

        ones         = np.ones((len(p_camera_unity), 1))
        P_hom        = np.hstack([p_camera_unity, ones])
        P_base_unity = (T_base_camera_unity @ P_hom.T).T[:, :3]"""

        T = np.array([
            [ 0.0,  1.0,  0.0,  0.52370],
            [ 0.0,  0.0, -1.0,  0.56416],
            [-1.0,  0.0,  0.0,  0.10915],
            [ 0.0,  0.0,  0.0,  1.0]
        ])

        ones = np.ones((len(p_camera_unity), 1))
        P_h  = np.hstack([p_camera_unity, ones])

        P_base_unity = (T @ P_h.T).T[:, :3]



        # Step 3: base Unity → base ROS
        points_ros = np.zeros_like(P_base_unity)
        points_ros[:, 0] = -P_base_unity[:, 2]  # x_ros = -z_unity
        points_ros[:, 1] =  P_base_unity[:, 0]  # y_ros =  x_unity
        points_ros[:, 2] =  P_base_unity[:, 1]  # z_ros =  y_unity

        return points_ros

    def genera_orientamenti(self, P):
        """
        Genera orientamenti con:
        - X = direzione di avanzamento (tangente)
        - Z = verso il basso (perpendicolare alla superficie)
        - Y = completa la terna ortonormale
        
        Garantisce che X sia ESATTAMENTE la tangente
        """
        R_list = []
        
        for k in range(len(P) - 1):
            # ========== X: TANGENTE (direzione avanzamento) ==========
            tang = P[k+1] - P[k]
            
            if np.linalg.norm(tang) > 1e-6:
                x_ee = tang / np.linalg.norm(tang)
            else:
                # Fallback: mantieni orientamento precedente
                if len(R_list) > 0:
                    x_ee = R_list[-1][:, 0]
                else:
                    x_ee = np.array([1.0, 0.0, 0.0])
            
            # ========== Z: FISSO VERSO IL BASSO ==========
            z_ee = np.array([0.0, 0.0, -1.0])
            
            # ========== Y: PERPENDICOLARE A X E Z ==========
            # y = z × x (completa terna destra)
            y_ee = np.cross(z_ee, x_ee)
            
            if np.linalg.norm(y_ee) > 1e-6:
                y_ee = y_ee / np.linalg.norm(y_ee)
            else:
                # Singolarità: tang parallelo a Z
                # Usa Y perpendicolare a X nel piano XY
                if abs(x_ee[2]) < 0.99:  # Non verticale
                    y_ee = np.array([-x_ee[1], x_ee[0], 0.0])
                    y_ee = y_ee / np.linalg.norm(y_ee)
                else:
                    y_ee = np.array([0.0, 1.0, 0.0])
            
            # ========== RICALCOLA Z per garantire ortonormalità ==========
            # (necessario se c'era singolarità)
            z_ee = np.cross(x_ee, y_ee)
            z_ee = z_ee / np.linalg.norm(z_ee)
            
            # Matrice di rotazione [x | y | z]
            R = np.column_stack((x_ee, y_ee, z_ee))
            R_list.append(R)
        
        # Ultimo orientamento = penultimo
        R_list.append(R_list[-1])


        # ── Filtro passa basso sui quaternioni ──────────────────────────
        quats = np.array([
            Rotation.from_matrix(R).as_quat() for R in R_list
        ])  # shape (N, 4) — formato [x, y, z, w]

        # Assicura continuità: evita flip di segno tra quaternioni consecutivi
        # (q e -q rappresentano la stessa rotazione ma il filtro li tratterebbe diversamente)
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i-1]) < 0:
                quats[i] = -quats[i]

        # Filtro passa basso: media mobile su finestra scorrevole
        alpha = 0.2  # ← regola qui: più basso = più smussato (0.05÷0.3)
        quats_filtered = quats.copy()
        for i in range(1, len(quats)):
            quats_filtered[i] = (1 - alpha) * quats_filtered[i-1] + alpha * quats[i]
            # Rinormalizza (la media lineare non preserva la norma unitaria)
            quats_filtered[i] /= np.linalg.norm(quats_filtered[i])

        # Riconverti in matrici
        R_list_filtered = [
            Rotation.from_quat(q).as_matrix() for q in quats_filtered
        ]

        return R_list_filtered



    def calcola_lunghezza_spline_semplice(self, sx, sy, sz):
        """
        Versione semplice: campiona 1000 punti uniformemente
        """
        s_sample = np.linspace(0, 1, 1000)
        
        x = sx(s_sample)
        y = sy(s_sample)
        z = sz(s_sample)
        
        # Somma distanze tra punti consecutivi
        distances = np.sqrt(
            np.diff(x)**2 + 
            np.diff(y)**2 + 
            np.diff(z)**2
        )
        
        L_total = np.sum(distances)
        
        return L_total
    
    def genera_profilo_trapezoidale_spazio(self, L_total, v_max, a_max):
        """
        Genera profilo trapezoidale NELLO SPAZIO (non nel tempo)
        
        Args:
            L_total: lunghezza percorso [m]
            v_max: velocità massima cartesiana [m/s]
            a_max: accelerazione massima [m/s²]
        
        Returns:
            s_array: posizioni lungo curva [m]
            v_array: velocità ad ogni posizione [m/s]
        """
        # Spazio necessario per accelerare/decelerare
        s_accel = v_max**2 / (2 * a_max)
        s_decel = s_accel
        
        # Verifica se c'è spazio per raggiungere v_max
        s_total_ramps = s_accel + s_decel
        
        if s_total_ramps > L_total:

            '''#MANTENGO ACCELERAZIONE E ABBASSO V
            # Profilo triangolare (non raggiunge v_max)
            v_peak = np.sqrt(a_max * L_total)
            s_accel = L_total / 2
            s_coast = 0
            s_decel = L_total / 2
            
            self.get_logger().info(f"⚠️  Profilo triangolare: v_peak={v_peak*1000:.1f} mm/s")'''

            #RIDUCO accelerazione per raggiungere esattamente v_max
            a_max = v_max**2 / L_total

            s_accel = L_total / 2
            s_decel = L_total / 2
            s_coast = 0
            v_peak = v_max

            self.get_logger().info(f"⚠️ Profilo limite: a_max ridotta = {a_max:.3f}")
        else:
            # Profilo trapezoidale normale
            v_peak = v_max
            s_coast = L_total - s_total_ramps
        
        # Numero di campioni spaziali
        n_samples = 1000
        s_array = np.linspace(0, L_total, n_samples)
        #print("S_ARRAY: ", s_array)
        v_array = np.zeros(n_samples)
        
        for i, s in enumerate(s_array):
            if s < s_accel:
                # Fase accelerazione: v² = 2·a·s
                v_array[i] = np.sqrt(2 * a_max * s)
            
            elif s < s_accel + s_coast:
                # Fase velocità costante
                v_array[i] = v_peak
            
            else:
                # Fase decelerazione: v² = 2·a·(s_remaining)
                s_remaining = L_total - s
                v_array[i] = np.sqrt(2 * a_max * s_remaining)
        
        return s_array, v_array


    def converti_spazio_in_tempo(self, s_array, v_array):
        """
        Converte profilo v(s) in traiettoria s(t)
        
        Returns:
            t_array: vettore tempo [s]
            s_time: posizioni campionate a dt costante
        """
        # Calcola tempo cumulativo
        t_cumulative = [0.0]
        
        for i in range(1, len(s_array)):
            ds = s_array[i] - s_array[i-1]
            v_avg = (v_array[i] + v_array[i-1]) / 2
            
            if v_avg > 1e-6:
                dt_segment = ds / v_avg
            else:
                dt_segment = 0.0
            
            t_cumulative.append(t_cumulative[-1] + dt_segment)
        
        t_cumulative = np.array(t_cumulative)
        #print("T_CUMULATIVE: ", t_cumulative)
        T_total = t_cumulative[-1]
        
        self.get_logger().info(f"Durata totale calcolata: {T_total:.3f} s")
        
        # ========== FIX: Gestisci boundary correttamente ==========
        from scipy.interpolate import interp1d
        
        # Crea interpolatore con extrapolation disabled
        s_of_t = interp1d(
            t_cumulative, 
            s_array, 
            kind='linear',
            bounds_error=False,      # ← Non dare errore fuori range
            fill_value=(s_array[0], s_array[-1])  # ← Usa valori ai bordi
        )
        
        # Genera vettore tempo SENZA superare T_total
        
        n_points = int(np.floor(T_total / self.dt)) + 1
        t_uniform = np.linspace(0, T_total, n_points)
        
        # ========== ALTERNATIVA: usa arange con clip ==========
        
        # Interpola
        s_time = s_of_t(t_uniform)
        
        # Assicurati che s sia nel range [0, s_max]
        s_time = np.clip(s_time, s_array[0], s_array[-1])
        
        self.get_logger().info(f"Punti generati: {len(t_uniform)}")
        self.get_logger().info(f"Range tempo: [{t_uniform[0]:.3f}, {t_uniform[-1]:.3f}] s")
        self.get_logger().info(f"Range spazio: [{s_time[0]:.3f}, {s_time[-1]:.3f}] m")
        
        return t_uniform, s_time
    

    def pubblica_traiettoria_desiderata(self):
        """
        Pubblica traiettoria come array di Pose
        """
        pose_array = PoseArray()
        pose_array.header.frame_id = "base_link"
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        # Aggiungi tutti i punti
        for i in range(self.N):
            pose = Pose()
            pose.position.x = self.x_d[i]
            pose.position.y = self.y_d[i]
            pose.position.z = self.z_d[i]
            
            # Orientamento (opzionale, usa identità se non serve)
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            
            pose_array.poses.append(pose)
        
        self.pub_trajectory_desired.publish(pose_array)
        self.get_logger().info(f"✅ Traiettoria pubblicata: {len(pose_array.poses)} punti")

    # =========================================================
    # CONTROL LOOP  ← cuore del controllore
    # =========================================================
    
    def control_loop(self):
        """
        Usa q_current dal robot
        """
        # Sicurezza: aspetta configurazione valida
        if self.q_current is None:
            self.get_logger().warn("Stato robot non ancora ricevuto...")
            return

        if self.idx >= self.N:
            self.get_logger().info("Traiettoria completata!")
            self.timer.cancel()
            return

        # Stato desiderato
        xd = np.array([self.x_d[self.idx],
                    self.y_d[self.idx],
                    self.z_d[self.idx]])
        vd = np.array([self.vx_d[self.idx],
                    self.vy_d[self.idx],
                    self.vz_d[self.idx]])
        Rd = self.R_d[self.idx]

        
        q = self.q_current.copy()

        # Cinematica diretta sullo stato REALE
        T_cur = self.fk(q)
        x_cur = T_cur[0:3, 3]
        R_cur = T_cur[0:3, 0:3]

        # Errore
        ep = xd - x_cur
        eo = self.orientation_error(R_cur, Rd)
        e  = np.hstack((ep, eo))

        # Velocità di riferimento
        v_ref = np.hstack((vd, np.zeros(3))) + self.K @ e

        # Jacobiano e legge di controllo
        J     = self.jacobian(q)
        J_inv = self.dls_inverse(J)
        dq    = J_inv @ v_ref

        # Saturazione velocità
        dq = np.clip(dq, -self.dq_max, self.dq_max)


        # Pubblica comando
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint',      'wrist_2_joint',        'wrist_3_joint'
        ]
        js.position = list(q + dq * self.dt)#.tolist()  # Posizione target
        #self.get_logger().info(f"{js.position}")
        js.velocity = list(dq)#dq.tolist()                   # Velocità target
        self.pub_joint.publish(js)

        # Log ogni secondo
        if self.idx % 50 == 0:
            self.get_logger().info(
                f"t={self.idx*self.dt:.2f}s | "
                f"|ep|={np.linalg.norm(ep)*1000:.2f}mm | "
                f"|eo|={np.linalg.norm(eo)*180/np.pi:.2f}°"
            )

        self.idx += 1
    
    def plot_trajectory_analysis(self):
        """
        Genera plot diagnostici completi della traiettoria end-effector:
        - Traiettoria 3D
        - Posizioni X, Y, Z nel tempo
        - Velocità cartesiane
        - Accelerazioni cartesiane
        - Velocità e accelerazione scalari
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Vettore tempo
        t = np.arange(0, self.N) * self.dt
        
        # ========== Calcola accelerazioni ==========
        # Accelerazione = derivata della velocità
        ax_d = np.gradient(self.vx_d, self.dt)
        ay_d = np.gradient(self.vy_d, self.dt)
        az_d = np.gradient(self.vz_d, self.dt)
        
        # Velocità e accelerazione scalari (norme)
        v_norm = np.sqrt(self.vx_d**2 + self.vy_d**2 + self.vz_d**2)
        a_norm = np.sqrt(ax_d**2 + ay_d**2 + az_d**2)
        
        # ========== CREA FIGURE ==========
        
        fig = plt.figure(figsize=(16, 12))
        
        # ========== 1. TRAIETTORIA 3D ==========
        ax1 = fig.add_subplot(3, 3, 1, projection='3d')
        
        # Traiettoria
        ax1.plot(self.x_d, self.y_d, self.z_d, 'b-', linewidth=2, label='Traiettoria')
        
        # Punto iniziale e finale
        ax1.scatter(self.x_d[0], self.y_d[0], self.z_d[0], 
                    c='green', s=100, marker='o', label='Start')
        ax1.scatter(self.x_d[-1], self.y_d[-1], self.z_d[-1], 
                    c='red', s=100, marker='x', label='End')
        

        ax1.scatter(self.P_completo[:, 0], self.P_completo[:, 1], self.P_completo[:, 2],
                c='red', s=80, marker='o', edgecolors='darkred', 
                linewidths=2, label='Waypoints', zorder=5)
        
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_zlabel('Z [m]')
        ax1.set_title('Traiettoria 3D End-Effector')
        ax1.legend()
        ax1.grid(True)
        
        # ========== 2. POSIZIONI NEL TEMPO ==========
        ax2 = fig.add_subplot(3, 3, 2)
        ax2.plot(t, self.x_d * 1000, 'r-', linewidth=1.5, label='X')
        ax2.plot(t, self.y_d * 1000, 'g-', linewidth=1.5, label='Y')
        ax2.plot(t, self.z_d * 1000, 'b-', linewidth=1.5, label='Z')
        ax2.set_xlabel('Tempo [s]')
        ax2.set_ylabel('Posizione [mm]')
        ax2.set_title('Posizioni Cartesiane')
        ax2.legend()
        ax2.grid(True)
        
        # ========== 3. VELOCITÀ COMPONENTI ==========
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.plot(t, self.vx_d * 1000, 'r-', linewidth=1.5, label='Vx')
        ax3.plot(t, self.vy_d * 1000, 'g-', linewidth=1.5, label='Vy')
        ax3.plot(t, self.vz_d * 1000, 'b-', linewidth=1.5, label='Vz')
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Tempo [s]')
        ax3.set_ylabel('Velocità [mm/s]')
        ax3.set_title('Velocità Cartesiane (componenti)')
        ax3.legend()
        ax3.grid(True)
        
        # ========== 4. VELOCITÀ SCALARE (NORMA) ==========
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.plot(t, v_norm * 1000, 'b-', linewidth=2)
        ax4.fill_between(t, 0, v_norm * 1000, alpha=0.3)
        ax4.set_xlabel('Tempo [s]')
        ax4.set_ylabel('Velocità [mm/s]')
        ax4.set_title('Velocità Scalare ||v||')
        ax4.grid(True)
        
        # Statistiche
        v_max = np.max(v_norm) * 1000
        v_avg = np.mean(v_norm) * 1000
        ax4.axhline(y=v_max, color='r', linestyle='--', linewidth=1, 
                    label=f'Max: {v_max:.1f} mm/s')
        ax4.axhline(y=v_avg, color='orange', linestyle='--', linewidth=1, 
                    label=f'Avg: {v_avg:.1f} mm/s')
        ax4.legend()
        
        # ========== 5. ACCELERAZIONI COMPONENTI ==========
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.plot(t, ax_d, 'r-', linewidth=1.5, label='ax')
        ax5.plot(t, ay_d, 'g-', linewidth=1.5, label='ay')
        ax5.plot(t, az_d, 'b-', linewidth=1.5, label='az')
        ax5.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax5.set_xlabel('Tempo [s]')
        ax5.set_ylabel('Accelerazione [m/s²]')
        ax5.set_title('Accelerazioni Cartesiane (componenti)')
        ax5.legend()
        ax5.grid(True)
        
        # ========== 6. ACCELERAZIONE SCALARE ==========
        ax6 = fig.add_subplot(3, 3, 6)
        ax6.plot(t, a_norm, 'r-', linewidth=2)
        ax6.fill_between(t, 0, a_norm, alpha=0.3, color='red')
        ax6.set_xlabel('Tempo [s]')
        ax6.set_ylabel('Accelerazione [m/s²]')
        ax6.set_title('Accelerazione Scalare ||a||')
        ax6.grid(True)
        
        # Statistiche
        a_max = np.max(a_norm)
        a_avg = np.mean(a_norm)
        ax6.axhline(y=a_max, color='darkred', linestyle='--', linewidth=1,
                    label=f'Max: {a_max:.3f} m/s²')
        ax6.axhline(y=a_avg, color='orange', linestyle='--', linewidth=1,
                    label=f'Avg: {a_avg:.3f} m/s²')
        ax6.legend()
        
        # ========== 7. TRAIETTORIA XY (vista dall'alto) ==========
        ax7 = fig.add_subplot(3, 3, 7)
        
        # Colora per velocità
        scatter = ax7.scatter(self.x_d * 1000, self.y_d * 1000, 
                            c=v_norm * 1000, cmap='viridis', 
                            s=2, alpha=0.8)
        
        # Waypoints
        ax7.scatter(self.x_d[0] * 1000, self.y_d[0] * 1000, 
                c='green', s=100, marker='o', zorder=5, label='Start')
        ax7.scatter(self.x_d[-1] * 1000, self.y_d[-1] * 1000, 
                c='red', s=100, marker='x', zorder=5, label='End')
        
        ax7.set_xlabel('X [mm]')
        ax7.set_ylabel('Y [mm]')
        ax7.set_title('Traiettoria XY (colorata per velocità)')
        ax7.set_aspect('equal')
        ax7.grid(True)
        ax7.legend()
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax7)
        cbar.set_label('Velocità [mm/s]')
        
        # ========== 8. PROFILO SPAZIO-VELOCITÀ ==========
        ax8 = fig.add_subplot(3, 3, 8)
        
        # Distanza percorsa cumulativa
        distances = np.zeros(self.N)
        for i in range(1, self.N):
            dx = self.x_d[i] - self.x_d[i-1]
            dy = self.y_d[i] - self.y_d[i-1]
            dz = self.z_d[i] - self.z_d[i-1]
            distances[i] = distances[i-1] + np.sqrt(dx**2 + dy**2 + dz**2)
        
        distances *= 1000  # Converti in mm
        
        ax8.plot(distances, v_norm * 1000, 'b-', linewidth=2)
        ax8.fill_between(distances, 0, v_norm * 1000, alpha=0.3)
        ax8.set_xlabel('Distanza percorsa [mm]')
        ax8.set_ylabel('Velocità [mm/s]')
        ax8.set_title('Velocità vs Spazio percorso')
        ax8.grid(True)
        
        # ========== 9. STATISTICHE ==========
        ax9 = fig.add_subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calcola statistiche
        total_distance = distances[-1]
        total_time = t[-1]
        v_avg_space = total_distance / (total_time * 1000)  # mm/s
        
        stats_text = f"""
        STATISTICHE TRAIETTORIA
        
        Durata totale:      {total_time:.2f} s
        Distanza totale:    {total_distance:.1f} mm
        
        Velocità:
        - Massima:        {v_max:.1f} mm/s
        - Media (tempo):  {v_avg:.1f} mm/s
        - Media (spazio): {v_avg_space:.1f} mm/s
        
        Accelerazione:
        - Massima:        {a_max:.3f} m/s²
        - Media:          {a_avg:.3f} m/s²
        
        Punti traiettoria:  {self.N}
        Frequenza:          {1/self.dt:.0f} Hz
        """
        
        ax9.text(0.1, 0.5, stats_text, 
                fontsize=11, 
                verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # ========== LAYOUT E SALVATAGGIO ==========
        plt.tight_layout()
        
        # Salva
        filename = '/tmp/trajectory_analysis.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        self.get_logger().info(f"Plot salvato in: {filename}")
        
        # Mostra (commenta se non hai display)
        plt.show()

    
    def plot_3d_trajectory_with_waypoints(self, P_waypoints):
        """
        Plot 3D della traiettoria con waypoints originali evidenziati
        
        Args:
            P_waypoints: array numpy (N, 3) dei waypoints originali
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # ========== TRAIETTORIA INTERPOLATA ==========
        ax.plot(self.x_d, self.y_d, self.z_d, 
                'b-', linewidth=2, alpha=0.7, label='Traiettoria interpolata')
        
        # ========== WAYPOINTS ORIGINALI ==========
        ax.scatter(P_waypoints[:, 0], P_waypoints[:, 1], P_waypoints[:, 2],
                c='red', s=80, marker='o', edgecolors='darkred', 
                linewidths=2, label='Waypoints', zorder=5)
        
        # Etichette sui waypoints
        for i, point in enumerate(P_waypoints):
            ax.text(point[0], point[1], point[2], 
                    f'  W{i}', fontsize=9, color='darkred', 
                    fontweight='bold')
        
        # ========== PUNTO INIZIALE E FINALE ==========
        ax.scatter(self.x_d[0], self.y_d[0], self.z_d[0], 
                c='green', s=150, marker='o', edgecolors='darkgreen',
                linewidths=3, label='Start', zorder=10)
        
        ax.scatter(self.x_d[-1], self.y_d[-1], self.z_d[-1], 
                c='orange', s=150, marker='X', edgecolors='darkorange',
                linewidths=3, label='End', zorder=10)
        
        # ========== LINEE DI CONNESSIONE WAYPOINTS ==========
        # Mostra il percorso originale tra waypoints
        ax.plot(P_waypoints[:, 0], P_waypoints[:, 1], P_waypoints[:, 2],
                'r--', linewidth=1, alpha=0.4, label='Percorso waypoints')
        
        # ========== GRIGLIA DI RIFERIMENTO ==========
        # Proietta traiettoria sui piani
        ax.plot(self.x_d, self.y_d, np.min(self.z_d) * np.ones_like(self.z_d),
                'gray', linewidth=0.5, alpha=0.3)  # Proiezione XY
        
        # ========== ASSI E LABELS ==========
        ax.set_xlabel('X [m]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y [m]', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z [m]', fontsize=12, fontweight='bold')
        ax.set_title('Traiettoria 3D con Waypoints', fontsize=14, fontweight='bold')
        
        # ========== LEGENDA ==========
        ax.legend(loc='upper left', fontsize=10)
        
        # ========== GRIGLIA ==========
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # ========== STATISTICHE ==========
        total_waypoints = len(P_waypoints)
        total_points = self.N
        
        # Aggiungi box con info
        info_text = f"""Waypoints: {total_waypoints}
        Punti interpolati: {total_points}
        Lunghezza: {self._calculate_path_length():.3f} m"""
        
        ax.text2D(0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # ========== SALVA ==========
        plt.tight_layout()
        #filename = '/tmp/trajectory_3d_waypoints.png'
        #plt.savefig(filename, dpi=200, bbox_inches='tight')
        #self.get_logger().info(f"📊 Plot 3D salvato in: {filename}")
        
        plt.show()


    def _calculate_path_length(self):
        """Calcola lunghezza totale del percorso"""
        length = 0
        for i in range(1, self.N):
            dx = self.x_d[i] - self.x_d[i-1]
            dy = self.y_d[i] - self.y_d[i-1]
            dz = self.z_d[i] - self.z_d[i-1]
            length += np.sqrt(dx**2 + dy**2 + dz**2)
        return length


def main(args=None):
    rclpy.init(args=args)
    node = KinematicController()
    rclpy.spin(node)
    rclpy.shutdown()