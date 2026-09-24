# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from builtin_interfaces.msg import Duration
# from std_msgs.msg import Float64MultiArray
# import numpy as np
# import sys



# class HomeJointController(Node):
#     def __init__(self):
#         super().__init__("home_joint_controller")

#         # Publisher aggiornato
#         """self.pub_joint = self.create_publisher(
#             JointTrajectory,
#             '/scaled_joint_trajectory_controller/joint_trajectory',
#             10
#         )"""

#         self.pub_joint = self.create_publisher(
#             Float64MultiArray,
#             '/forward_position_controller/commands',
#             10
#         )

#         # Subscriber stato
#         self.q_current = None
#         self.sub_joint = self.create_subscription(
#             JointState, '/joint_states', self.joint_state_callback, 10
#         )

#         # Guadagno feedback
#         self.Kp = 3.0

#         # Configurazioni
#         self.configs = {
#             'home': np.array([np.pi+(np.pi/2), -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0]),
#             'up': np.array([0.0, -np.pi/2, 0.0, -np.pi/2, -np.pi/2, 0.0]),
#             'camera' : np.array([np.deg2rad(89.84), np.deg2rad(-78.81), np.deg2rad(3.45), np.deg2rad(-23.04), np.deg2rad(-98.58), np.deg2rad(0)]),
#             'ortogonale' : np.array([np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
#         }

#         # Parametri DH UR5
#         self.a     = [0,       -0.425,   -0.39225, 0,       0,       0      ]
#         self.alpha = [np.pi/2,  0,        0,        np.pi/2, -np.pi/2, 0     ]
#         self.d     = [0.089159, 0,        0,        0.10915, 0.09465, 0.0823 ]

#         # Traiettoria
#         self.trajectory = []
#         self.traj_idx = 0
#         self.timer = None
#         self.dt = 0.025

#         self.joint_names = [
#             'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#         ]

#         self.get_logger().info("HomeJointController avviato!")


#     def joint_state_callback(self, msg: JointState):
#         q = np.zeros(6)
#         for i, name in enumerate(self.joint_names):
#             if name in msg.name:
#                 idx = msg.name.index(name)
#                 q[i] = msg.position[idx]

#         if self.q_current is None:
#             self.get_logger().info("Stato iniziale ricevuto")

#         self.q_current = q


#     def genera_profilo_trapezoidale_normalizzato(self, duration, frac_acc=0.25):
#         """
#         Genera un profilo trapezoidale scalare s(t) che va da 0 a 1 in 'duration' secondi,
#         con velocità nulla a inizio e fine. frac_acc è la frazione di tempo dedicata
#         all'accelerazione (e altrettanta alla decelerazione); il resto è crociera a v costante.

#         Ritorna:
#             t: array dei tempi campionati (0..duration, passo self.dt)
#             s: posizione normalizzata (0..1)
#         """
#         t = np.arange(0, duration + self.dt, self.dt)

#         t_acc = frac_acc * duration
#         t_flat = duration - 2 * t_acc

#         if t_flat < 0:
#             # frac_acc troppo alto: niente crociera, profilo triangolare puro
#             t_acc = duration / 2.0
#             t_flat = 0.0

#         v_cruise = 1.0 / (duration - t_acc)   # area totale sotto il trapezio = 1 (spostamento normalizzato)
#         a = v_cruise / t_acc if t_acc > 1e-9 else 0.0

#         s = np.zeros_like(t)
#         for i, ti in enumerate(t):
#             if ti < t_acc:
#                 # fase di accelerazione
#                 s[i] = 0.5 * a * ti**2
#             elif ti < t_acc + t_flat:
#                 # fase di crociera
#                 s[i] = 0.5 * a * t_acc**2 + v_cruise * (ti - t_acc)
#             else:
#                 # fase di decelerazione (simmetrica)
#                 td = duration - ti
#                 td = max(td, 0.0)
#                 s[i] = 1.0 - 0.5 * a * td**2

#         s = np.clip(s, 0.0, 1.0)
#         return t, s


#     def plan_to_config(self, q_target, duration=17.0, frac_acc=0.25):
#         if self.q_current is None:
#             self.get_logger().error("Stato non disponibile!")
#             return False

#         self.get_logger().info(f"Pianificazione movimento trapezoidale (durata {duration}s)")

#         t, s = self.genera_profilo_trapezoidale_normalizzato(duration, frac_acc=frac_acc)

#         self.trajectory = []
#         for si in s:
#             q = self.q_current + si * (q_target - self.q_current)
#             self.trajectory.append(q)

#         self.get_logger().info(f"{len(self.trajectory)} punti generati")
#         return True


#     def execute(self):
#         if not self.trajectory:
#             self.get_logger().error("Nessuna traiettoria!")
#             return

#         self.traj_idx = 0
#         self.get_logger().info("Esecuzione con correzione errore...")

#         if self.timer:
#             self.timer.cancel()

#         self.timer = self.create_timer(self.dt, self.trajectory_callback_with_feedback)


#     def trajectory_callback_with_feedback(self):
#         if self.traj_idx >= len(self.trajectory):
#             if self.q_current is not None:
#                 q_target = self.trajectory[-1]
#                 error = q_target - self.q_current
#                 error_deg = np.rad2deg(np.abs(error))
#                 self.get_logger().info("Movimento completato!")
#                 self.get_logger().info(f"   Errore finale per giunto: {error_deg}")
#                 self.get_logger().info(f"   Errore totale: {np.linalg.norm(error_deg):.2f}°")

#             self.timer.cancel()
#             return

#         q_desired = self.trajectory[self.traj_idx]

#         if self.q_current is not None:
#             error = q_desired - self.q_current
#             q_cmd = q_desired + self.Kp * error * self.dt

#             if self.traj_idx % 50 == 0:
#                 error_norm = np.linalg.norm(error) * 180/np.pi
#                 progress = (self.traj_idx / len(self.trajectory)) * 100
#                 self.get_logger().info(f"  {progress:.0f}% | Errore: {error_norm:.2f}°")
#         else:
#             q_cmd = q_desired

#         """# Pubblica come JointTrajectory
#         traj = JointTrajectory()
#         traj.header.stamp = self.get_clock().now().to_msg()
#         traj.joint_names = self.joint_names

#         point = JointTrajectoryPoint()
#         point.positions = q_cmd.tolist()
#         point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # aggiungi questa riga
#         point.time_from_start = Duration(sec=0, nanosec=int(self.dt * 1e9))

#         traj.points = [point]
#         self.pub_joint.publish(traj)"""

#         msg = Float64MultiArray()
#         msg.data = q_cmd.tolist()
#         self.pub_joint.publish(msg)

#         self.traj_idx += 1


#     def fk_parziale(self, q, fino_a_giunto: int) -> np.ndarray:
#         T = np.eye(4)
#         for i in range(fino_a_giunto):
#             ct = np.cos(q[i]); st = np.sin(q[i])
#             ca = np.cos(self.alpha[i]); sa = np.sin(self.alpha[i])
#             T_i = np.array([
#                 [ct,   -st*ca,  st*sa,  self.a[i]*ct],
#                 [st,    ct*ca, -ct*sa,  self.a[i]*st],
#                 [0,     sa,     ca,     self.d[i]    ],
#                 [0,     0,      0,      1            ]
#             ])
#             T = T @ T_i
#         return T


#     def go_to_named_config(self, name, duration=17.0):
#         if name not in self.configs:
#             self.get_logger().error(f"Config '{name}' non trovata!")
#             return False

#         if self.plan_to_config(self.configs[name], duration):
#             self.execute()
#             return True
#         return False


# def main(args=None):
#     rclpy.init(args=args)
#     node = HomeJointController()

#     import time
#     for _ in range(10):
#         rclpy.spin_once(node, timeout_sec=0.2)
#         if node.q_current is not None:
#             break

#     if node.q_current is None:
#         node.get_logger().error("Stato non ricevuto!")
#         rclpy.shutdown()
#         return

#     if len(sys.argv) > 1:
#         config_name = sys.argv[1]
#         node.go_to_named_config(config_name, duration=17.0)
#     else:
#         node.go_to_named_config('home', duration=17.0)

#     rclpy.spin(node)
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
import numpy as np
import sys
import signal


class HomeJointController(Node):
    def __init__(self):
        super().__init__("home_joint_controller")

        # Action client verso lo scaled_joint_trajectory_controller
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        # Subscriber stato
        self.q_current = None
        self.sub_joint = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Configurazioni
        self.configs = {
            'home': np.array([np.pi + (np.pi/2), -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0]),
            'up': np.array([0.0, -np.pi/2, 0.0, -np.pi/2, -np.pi/2, 0.0]),
            'camera': np.array([np.deg2rad(89.84), np.deg2rad(-78.81), np.deg2rad(3.45),
                                 np.deg2rad(-23.04), np.deg2rad(-98.58), np.deg2rad(0)]),
            'ortogonale': np.array([np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0]),
            'finale': np.array([np.deg2rad(90.03), np.deg2rad(-88.37), np.deg2rad(23.04),
                                 np.deg2rad(-37.65), np.deg2rad(-89.64), np.deg2rad(0.02)])
        }

        # Parametri DH UR5 (non usati direttamente qui, ma lasciati se ti servono altrove)
        self.a     = [0,       -0.425,   -0.39225, 0,       0,       0      ]
        self.alpha = [np.pi/2,  0,        0,        np.pi/2, -np.pi/2, 0     ]
        self.d     = [0.089159, 0,        0,        0.10915, 0.09465, 0.0823 ]

        self.trajectory = []
        self.dt = 0.025

        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        self._current_goal_handle = None

        self.get_logger().info("HomeJointController avviato!")

    # ------------------------------------------------------------------
    # Stato robot
    # ------------------------------------------------------------------
    def joint_state_callback(self, msg: JointState):
        q = np.zeros(6)
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]

        if self.q_current is None:
            self.get_logger().info("Stato iniziale ricevuto")

        self.q_current = q

    # ------------------------------------------------------------------
    # Pianificazione: profilo trapezoidale normalizzato + interpolazione lineare joint-space
    # ------------------------------------------------------------------
    def genera_profilo_trapezoidale_normalizzato(self, duration, frac_acc=0.25):
        t = np.arange(0, duration + self.dt, self.dt)

        t_acc = frac_acc * duration
        t_flat = duration - 2 * t_acc

        if t_flat < 0:
            t_acc = duration / 2.0
            t_flat = 0.0

        v_cruise = 1.0 / (duration - t_acc)
        a = v_cruise / t_acc if t_acc > 1e-9 else 0.0

        s = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti < t_acc:
                s[i] = 0.5 * a * ti**2
            elif ti < t_acc + t_flat:
                s[i] = 0.5 * a * t_acc**2 + v_cruise * (ti - t_acc)
            else:
                td = max(duration - ti, 0.0)
                s[i] = 1.0 - 0.5 * a * td**2

        s = np.clip(s, 0.0, 1.0)
        return t, s

    def plan_to_config(self, q_target, duration=17.0, frac_acc=0.25):
        if self.q_current is None:
            self.get_logger().error("Stato non disponibile!")
            return False

        self.get_logger().info(f"Pianificazione movimento trapezoidale (durata {duration}s)")

        t, s = self.genera_profilo_trapezoidale_normalizzato(duration, frac_acc=frac_acc)

        self.trajectory = []
        for si in s:
            q = self.q_current + si * (q_target - self.q_current)
            self.trajectory.append(q)

        self.get_logger().info(f"{len(self.trajectory)} punti generati")
        return True

    # ------------------------------------------------------------------
    # Esecuzione via action FollowJointTrajectory
    # ------------------------------------------------------------------
    def execute(self):
        if not self.trajectory:
            self.get_logger().error("Nessuna traiettoria!")
            return

        self.get_logger().info("In attesa dell'action server...")
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server non disponibile!")
            return

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        points = []
        for k, q in enumerate(self.trajectory):
            point = JointTrajectoryPoint()
            point.positions = q.tolist()
            point.velocities = [0.0] * 6

            t_sec = (k + 1) * self.dt  # mai zero sul primo punto
            point.time_from_start = Duration(
                sec=int(t_sec),
                nanosec=int(round((t_sec - int(t_sec)) * 1e9))
            )
            points.append(point)

        traj.points = points

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj
        goal_msg.goal_time_tolerance = Duration(sec=5, nanosec=0)

        self.get_logger().info(
            f"Invio goal: {len(points)} punti, durata {len(points) * self.dt:.2f}s"
        )
        send_goal_future = self.action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        # Facoltativo: logga il progresso senza intasare la console
        pass

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(">>> GOAL RIFIUTATO dal controller! <<<")
            self._current_goal_handle = None
            return

        self._current_goal_handle = goal_handle
        self.get_logger().info("Goal ACCETTATO, esecuzione in corso...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f">>> Esecuzione terminata. error_code={result.error_code}, "
            f"error_string='{result.error_string}' <<<"
        )
        self._current_goal_handle = None

    # ------------------------------------------------------------------
    # Annullamento
    # ------------------------------------------------------------------
    def annulla_traiettoria(self):
        if self._current_goal_handle is not None:
            self.get_logger().info("Invio richiesta di cancellazione del goal...")
            cancel_future = self._current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_response_callback)
        else:
            self.get_logger().warn("Nessun goal attivo da cancellare.")

    def cancel_response_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info("Goal in fase di cancellazione confermata dal server.")
        else:
            self.get_logger().warn("Il server non ha confermato la cancellazione.")

    # ------------------------------------------------------------------
    # Cinematica diretta parziale (invariata, lasciata se ti serve altrove)
    # ------------------------------------------------------------------
    def fk_parziale(self, q, fino_a_giunto: int) -> np.ndarray:
        T = np.eye(4)
        for i in range(fino_a_giunto):
            ct = np.cos(q[i]); st = np.sin(q[i])
            ca = np.cos(self.alpha[i]); sa = np.sin(self.alpha[i])
            T_i = np.array([
                [ct,   -st*ca,  st*sa,  self.a[i]*ct],
                [st,    ct*ca, -ct*sa,  self.a[i]*st],
                [0,     sa,     ca,     self.d[i]    ],
                [0,     0,      0,      1            ]
            ])
            T = T @ T_i
        return T

    def go_to_named_config(self, name, duration=17.0):
        if name not in self.configs:
            self.get_logger().error(f"Config '{name}' non trovata!")
            return False

        if self.plan_to_config(self.configs[name], duration):
            self.execute()
            return True
        return False


def main(args=None):
    rclpy.init(args=args)
    node = HomeJointController()

    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.q_current is not None:
            break

    if node.q_current is None:
        node.get_logger().error("Stato non ricevuto!")
        rclpy.shutdown()
        return

    def stop_handler(sig, frame):
        node.get_logger().info("Ctrl+C ricevuto: annullo la traiettoria (se attiva)...")
        node.annulla_traiettoria()
        rclpy.spin_once(node, timeout_sec=1.0)
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_handler)

    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        node.go_to_named_config(config_name, duration=17.0)
    else:
        node.go_to_named_config('home', duration=17.0)

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()