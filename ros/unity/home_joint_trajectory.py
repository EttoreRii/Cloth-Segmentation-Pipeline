#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import sys


class HomeJointController(Node):
    """
    Pianificatore con FEEDBACK (closed-loop)
    - Genera traiettoria interpolata
    - Corregge errore in tempo reale
    - Garantisce precisione finale
    """

    def __init__(self):
        super().__init__("home_joint_controller")

        # Publisher
        self.pub_joint = self.create_publisher(JointState, '/joint_targets', 10)

        # Subscriber stato
        self.q_current = None
        self.sub_joint = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Guadagno feedback
        self.Kp = 3.0  # Proporzionale

        # Configurazioni
        self.configs = {
            'home': np.array([np.pi+(np.pi/2), -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0]), #np.zeros(6),
            'up': np.array([0.0, -np.pi/2, 0.0, -np.pi/2, -np.pi/2, 0.0]),
        }


        # Parametri DH UR5
        self.a     = [0,       -0.425,   -0.39225, 0,       0,       0      ]
        self.alpha = [np.pi/2,  0,        0,        np.pi/2, -np.pi/2, 0     ]
        self.d     = [0.089159, 0,        0,        0.10915, 0.09465, 0.0823 ]

        # Traiettoria
        self.trajectory = []
        self.traj_idx = 0
        self.timer = None
        self.dt = 0.03  # 50 Hz

        self.get_logger().info("HomeJointController avviato!")

    
    def joint_state_callback(self, msg: JointState):
        """Riceve stato corrente"""
        joint_order = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        q = np.zeros(6)
        for i, name in enumerate(joint_order):
            if name in msg.name:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]

        if self.q_current is None:
            self.get_logger().info(f"Stato iniziale ricevuto")
        
        self.q_current = q


    def plan_to_config(self, q_target, duration=17.0):
        """Genera traiettoria"""
        if self.q_current is None:
            self.get_logger().error("Stato non disponibile!")
            return False

        self.get_logger().info(f"Pianificazione movimento (durata {duration}s)")

        T = self.fk_parziale(q_target, fino_a_giunto=5)
        print("T_base_wrist2:\n", T)
        print("Posizione wrist2 in base:", T[0:3, 3])

        # Time scaling quintico
        t = np.arange(0, duration + self.dt, self.dt)
        tau = t / duration
        s = 10*tau**3 - 15*tau**4 + 6*tau**5

        # Interpola
        self.trajectory = []
        for si in s:
            q = self.q_current + si * (q_target - self.q_current)
            self.trajectory.append(q)

        self.get_logger().info(f"{len(self.trajectory)} punti generati")
        return True


    def execute(self):
        """Esegue con feedback"""
        if not self.trajectory:
            self.get_logger().error("Nessuna traiettoria!")
            return

        self.traj_idx = 0
        self.get_logger().info("Esecuzione con correzione errore...")

        if self.timer:
            self.timer.cancel()
        
        self.timer = self.create_timer(self.dt, self.trajectory_callback_with_feedback)


    def trajectory_callback_with_feedback(self):
        """
        Pubblica comando con FEEDBACK
        
        Legge:  q_current (stato reale)
        Vuole:  q_desired (dalla traiettoria)
        Manda:  q_cmd = q_desired + correzione
        """
        if self.traj_idx >= len(self.trajectory):
            # Verifica precisione finale
            if self.q_current is not None:
                q_target = self.trajectory[-1]
                error = q_target - self.q_current
                error_deg = np.rad2deg(np.abs(error))
                
                self.get_logger().info(
                    f"Movimento completato!"
                )
                self.get_logger().info(
                    f"   Errore finale per giunto: {error_deg}"
                )
                self.get_logger().info(
                    f"   Errore totale: {np.linalg.norm(error_deg):.2f}°"
                )
            
            self.timer.cancel()
            return

        # Configurazione desiderata (feedforward)
        q_desired = self.trajectory[self.traj_idx]

        # Calcola comando con feedback
        if self.q_current is not None:
            # Errore
            error = q_desired - self.q_current
            
            # Comando = desiderato + correzione proporzionale
            q_cmd = q_desired + self.Kp * error * self.dt
            
            # Log ogni secondo
            if self.traj_idx % 50 == 0:
                error_norm = np.linalg.norm(error) * 180/np.pi
                progress = (self.traj_idx / len(self.trajectory)) * 100
                self.get_logger().info(
                    f"  {progress:.0f}% | Errore: {error_norm:.2f}°"
                )
        else:
            # Fallback: solo feedforward se stato non disponibile
            q_cmd = q_desired

        # Pubblica
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        js.position = q_cmd.tolist()
        
        self.pub_joint.publish(js)
        self.traj_idx += 1
    

    def fk_parziale(self, q, fino_a_giunto: int) -> np.ndarray:
        """
        FK fino a un giunto specifico (0-indexed).
        fino_a_giunto=5 → fino a wrist_2
        fino_a_giunto=6 → fino al flange completo
        """
        T = np.eye(4)
        for i in range(fino_a_giunto):
            ct = np.cos(q[i]);  st = np.sin(q[i])
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
        """Vai a configurazione predefinita"""
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

    # Aspetta stato
    import time
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.q_current is not None:
            break
    
    if node.q_current is None:
        node.get_logger().error("Stato non ricevuto!")
        rclpy.shutdown()
        return

    # Esegui comando
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        node.go_to_named_config(config_name, duration=17.0)
    else:
        node.go_to_named_config('home', duration=17.0)

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()