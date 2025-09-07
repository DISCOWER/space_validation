import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product


# ================= Thruster Geometry =================
def _thruster_rotational_matrix(alpha):
    """Returns a 3x3 rotation matrix for a given yaw angle alpha."""
    return np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0.0],
            [np.sin(alpha), np.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def get_thruster_data(robot_type: str, side_length=0.30):
    """
    Defines and returns thruster positions and direction vectors for different robots.

    Args:
        robot_type: "bluerov", "atmos", or "astrobee"
        side_length: used for "atmos" and "astrobee" robots.

    Returns:
        tuple: (positions, directions, dimensions)
    """
    thruster_list = []

    if robot_type == "bluerov":
        # Base positions
        r1234 = np.array([0.156, 0.111, 0.085])
        r5678 = np.array([0.12, 0.218, 0.0])
        # Base orientation
        e1234 = np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0.0])
        # Rotation angles
        J3_r_angles = [0.0, 5.05, 1.91, np.pi, 0.0, 4.15, 1.01, np.pi]
        J3_e_angles = [0.0, np.pi / 2, 3 * np.pi / 2, np.pi]
        # Horizontal thrusters T1..T4
        for i in range(4):
            thruster_list.append(
                {
                    "r": np.dot(_thruster_rotational_matrix(J3_r_angles[i]), r1234),
                    "dir": np.dot(_thruster_rotational_matrix(J3_e_angles[i]), e1234),
                }
            )
        # Vertical thrusters T5..T8
        for i in range(4, 8):
            thruster_list.append(
                {
                    "r": np.dot(_thruster_rotational_matrix(J3_r_angles[i]), r5678),
                    "dir": np.array([0.0, 0.0, -1.0]),
                }
            )
        positions = np.array([t["r"] for t in thruster_list]).T
        directions = np.array([t["dir"] for t in thruster_list]).T
        return positions, directions, 3

    elif robot_type == "atmos":
        half = side_length / 2
        # Four corners of the square (x, y)
        corners = [
            np.array([half, half]),  # top-right
            np.array([-half, half]),  # top-left
            np.array([-half, -half]),  # bottom-left
            np.array([half, -half]),  # bottom-right
        ]
        # Each corner has 2 thrusters, orthogonal along edges, pushing outward
        for x, y in corners:
            # Thruster along x-axis
            thruster_list.append(
                {
                    "r": np.array([x, y], dtype=float),
                    "dir": np.array([np.sign(x), 0.0], dtype=float),
                }
            )
            # Thruster along y-axis
            thruster_list.append(
                {
                    "r": np.array([x, y], dtype=float),
                    "dir": np.array([0.0, np.sign(y)], dtype=float),
                }
            )
        positions = np.array([t["r"] for t in thruster_list]).T
        directions = np.array([t["dir"] for t in thruster_list]).T
        return positions, directions, 2

    elif robot_type == "astrobee":
        half = side_length / 2
        corners = list(product([-half, half], repeat=3))

        for x, y, z in corners:
            thruster_list.append(
                {"r": np.array([x, y, z]), "dir": np.array([np.sign(x), 0, 0])}
            )
            thruster_list.append(
                {"r": np.array([x, y, z]), "dir": np.array([0, np.sign(y), 0])}
            )
            thruster_list.append(
                {"r": np.array([x, y, z]), "dir": np.array([0, 0, np.sign(z)])}
            )

        positions = np.array([t["r"] for t in thruster_list]).T
        directions = np.array([t["dir"] for t in thruster_list]).T
        return positions, directions, 3


# ================= Approximate Inertia =================
def approx_inertia_box(m: float, L: float, W: float, H: float):
    """
    Approximates inertia of the vehicle by a rectangular prism (about CoM).
    Returns diagonal inertia [Ixx, Iyy, Izz] in kg*m^2.
    """
    Ixx = (1.0 / 12.0) * m * (W**2 + H**2)
    Iyy = (1.0 / 12.0) * m * (L**2 + H**2)
    Izz = (1.0 / 12.0) * m * (L**2 + W**2)
    return np.diag([Ixx, Iyy, Izz])


# ================= Polytope Computation =================
def get_thruster_bounds(robot_type: str):
    """Return per-thruster asymmetric force limits based on robot type."""
    if robot_type == "bluerov":
        T_forward_max = 51.5  # N
        T_reverse_max = -40.2  # N
        num_thrusters = 8
    elif robot_type == "atmos":
        # Example values for ATMOS thrusters (can be changed)
        T_forward_max = 2/3*1.075 #1.5  # N
        T_reverse_max = 0  # N
        num_thrusters = 8
    elif robot_type == "astrobee":
        # Example values for Astrobee thrusters (can be changed)
        T_forward_max = 0.75*2/3*1.075 #1.5  # N
        T_reverse_max = 0  # N
        num_thrusters = 24

    u_max = np.full(num_thrusters, T_forward_max)
    u_min = np.full(num_thrusters, T_reverse_max)
    return u_min, u_max


def get_h_representation(vertices):
    """Compute H-representation (A x <= b) from convex hull of vertices."""
    # Handle the 1D case separately as ConvexHull requires at least 2-D data
    if vertices.ndim == 2 and vertices.shape[1] == 1:
        min_val = np.min(vertices)
        max_val = np.max(vertices)
        A = np.array([[1.0], [-1.0]])
        b = np.array([max_val, -min_val])
        precision = 10
        return np.around(A, precision), np.around(b, precision)

    hull = ConvexHull(vertices)
    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1]
    precision = 10
    return np.around(A, precision), np.around(b, precision)


def compute_inscribed_ball(A, b):
    """Radius of largest ball centered at origin inside A x <= b."""
    if np.any(b < 0):
        print("Warning: Origin not inside polytope.")
        return 0.0
    face_distances = b / np.linalg.norm(A, axis=1)
    return np.min(face_distances)


# ================= Regular Polytopes =================
def regular_polytope_vertices(poly_type: str, radius: float, center: np.ndarray):
    """Generate vertices of a regular polytope inscribed in a sphere."""
    if center.shape[0] == 1:
        return np.array([[-radius], [radius]])

    if poly_type == "cube":
        if center.shape[0] == 3:
            r_prime = radius / np.sqrt(3)
            vertices = np.array(list(product([-r_prime, r_prime], repeat=3)))
        else:  # 2D case
            r_prime = radius / np.sqrt(2)
            vertices = np.array(list(product([-r_prime, r_prime], repeat=2)))
    elif poly_type == "octahedron":
        vertices = (
            np.array(
                [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
            )
            * radius
        )
    elif poly_type == "tetrahedron":
        a = 1 / np.sqrt(3)
        vertices = np.array([[a, a, a], [-a, -a, a], [-a, a, -a], [a, -a, -a]]) * radius
    elif poly_type == "icosahedron":
        phi = (1 + np.sqrt(5)) / 2
        vertices = []
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                vertices.append([0, s1, s2 * phi])
                vertices.append([s1, s2 * phi, 0])
                vertices.append([s1 * phi, 0, s2])
        vertices = np.array(vertices, float)
        vertices /= np.linalg.norm(vertices[0])
        vertices *= radius
    else:
        raise ValueError(f"Unknown polytope type {poly_type}")
    return vertices + center


# ================= Plotting =================
def plot_polytope(vertices, ax, title, color, radius=None, alpha=0.3):
    """Plot convex hull and optional inscribed sphere in 1D, 2D or 3D."""
    dimensions = vertices.shape[1]

    if dimensions == 3:
        # 3D Plotting
        hull = ConvexHull(vertices)
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="r", s=20)
        for simplex in hull.simplices:
            pts = vertices[simplex]
            ax.add_collection3d(Poly3DCollection([pts], alpha=alpha, facecolor=color))
        if radius is not None and radius > 0:
            u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 25j]
            x = radius * np.cos(u) * np.sin(v)
            y = radius * np.sin(u) * np.sin(v)
            z = radius * np.cos(v)
            ax.plot_surface(x, y, z, color="orange", alpha=0.2, linewidth=0)
        ax.set_xlabel("Fx/τx")
        ax.set_ylabel("Fy/τy")
        ax.set_zlabel("Fz/τz")

    elif dimensions == 2:
        # 2D Plotting
        hull = ConvexHull(vertices)
        for simplex in hull.simplices:
            pts = vertices[simplex]
            ax.plot(pts[:, 0], pts[:, 1], color=color)
            ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=alpha)
        ax.scatter(vertices[:, 0], vertices[:, 1], color="r", s=20, zorder=10)
        if radius is not None and radius > 0:
            circle = plt.Circle(
                (0, 0), radius, color="orange", fill=False, linestyle="--", linewidth=2
            )
            ax.add_patch(circle)
        ax.set_xlabel("Fx")
        ax.set_ylabel("Fy")
        ax.set_aspect("equal", adjustable="box")

    elif dimensions == 1:
        # 1D Plotting
        x_min, x_max = np.min(vertices), np.max(vertices)
        ax.plot([x_min, x_max], [0, 0], color=color, linewidth=5)
        ax.scatter([x_min, x_max], [0, 0], color="r", zorder=10)
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_xlabel("Torque (τz)")
        if radius is not None and radius > 0:
            ax.plot(
                [-radius, radius], [0, 0], color="orange", linestyle="--", linewidth=2
            )
            ax.scatter([-radius, radius], [0, 0], color="orange")

    ax.set_title(title)
    ax.grid(True)


# ====================== Helper functions ==================
def worst_gyro_norm(J: np.ndarray, W: float):
    """
    Given symmetric positive-definite inertia matrix J (3x3)
    and scalar bound W on ||omega||, return:
      - worst: max || omega x (J omega) || over ||omega|| <= W
      - omega_worst: one omega achieving it (norm W)
      - lambda_min, lambda_max: eigenvalues of J
    """
    # eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(J)  # sorted ascending
    lam_min, lam_max = eigvals[0], eigvals[-1]
    # worst-case norm
    worst = 0.5 * (lam_max - lam_min) * W**2

    # construct a witnessing omega: equal-energy combination of eigvec_min and eigvec_max
    e_min = eigvecs[:, 0]
    e_max = eigvecs[:, -1]
    v = e_min + e_max
    v = v / np.linalg.norm(v)  # unit
    omega_worst = W * v

    return worst, omega_worst, (lam_min, lam_max)


# ================= High-Level Function =================
def compute_polytope(
    robot_type: str, kind: str, poly_type: str, plot: bool, J=None, W_max=None
):
    """
    Compute H-representation of inscribed polytope.

    Args:
        robot_type: "bluerov", "atmos", or "astrobee"
        kind: "force" or "torque"
        poly_type: type of inscribed regular polytope ("cube", "tetrahedron", "icosahedron", etc.)
        plot: if True, plot the polytope and inscribed regular polytope
        J: inertia matrix (3x3), required if kind="torque" with gyro reduction
        W_max: angular speed bound (rad/s), required if kind="torque" with gyro reduction

    Returns:
        tuple: A, b, radius_nominal, radius_eff, gyro_bound
    """
    positions, directions, dimensions = get_thruster_data(robot_type)
    u_min, u_max = get_thruster_bounds(robot_type)

    # Generate all vertices (product of bounds)
    thrust_values = list(product(*zip(u_min, u_max)))
    U = np.array(thrust_values)

    # Calculate force vertices
    F_vertices = U @ directions.T

    # Calculate torque vertices (adjusted for dimensions)
    if dimensions == 2:
        T = np.zeros((1, directions.shape[1]))
        for i in range(directions.shape[1]):
            T[0, i] = (
                positions[0, i] * directions[1, i] - positions[1, i] * directions[0, i]
            )
        Tau_vertices = U @ T.T
    else:
        T = np.zeros((3, directions.shape[1]))
        for i in range(directions.shape[1]):
            T[:, i] = np.cross(positions[:, i], directions[:, i])
        Tau_vertices = U @ T.T

    vertices = F_vertices if kind == "force" else Tau_vertices

    A, b = get_h_representation(vertices)
    radius_nominal = compute_inscribed_ball(A, b)

    # For torque polytope, reduce radius by worst-case gyro effect if J and W_max are provided
    radius_eff = radius_nominal
    gyro_bound = None
    if kind == "torque" and dimensions == 3 and J is not None and W_max is not None:
        gyro_bound, _, _ = worst_gyro_norm(J, W_max)
        radius_eff = max(0.0, radius_nominal - gyro_bound)

    # Choose which radius to inscribe (effective if torque)
    inscribed_vertices = regular_polytope_vertices(
        poly_type, radius=radius_eff, center=np.zeros(dimensions)
    )
    A_inscribed, b_inscribed = get_h_representation(inscribed_vertices)

    if plot:
        fig = plt.figure(figsize=(8, 8))
        if dimensions == 3:
            ax = fig.add_subplot(111, projection="3d")
        elif dimensions == 2:
            ax = fig.add_subplot(111)
        else:  # 1D
            ax = fig.add_subplot(111)

        # Plot full polytope
        plot_polytope(
            vertices,
            ax,
            f"{robot_type.capitalize()} {kind.capitalize()} Polytope",
            "cyan",
            radius=radius_nominal,  # 🔹 show nominal ball
        )

        # Plot inscribed regular polytope
        plot_polytope(
            inscribed_vertices,
            ax,
            f"{robot_type.capitalize()} {kind.capitalize()} Polytope",
            "blue",
            alpha=0.6,
        )

        # If torque with gyro effect, also plot reduced sphere
        if kind == "torque" and gyro_bound is not None:
            if dimensions == 3:
                u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 25j]
                x = radius_eff * np.cos(u) * np.sin(v)
                y = radius_eff * np.sin(u) * np.sin(v)
                z = radius_eff * np.cos(v)
                ax.plot_surface(x, y, z, color="red", alpha=0.2, linewidth=0)
            elif dimensions == 2:
                circle = plt.Circle(
                    (0, 0),
                    radius_eff,
                    color="red",
                    fill=False,
                    linestyle="--",
                    linewidth=2,
                )
                ax.add_patch(circle)
            elif dimensions == 1:
                ax.plot(
                    [-radius_eff, radius_eff],
                    [0, 0],
                    color="red",
                    linestyle="--",
                    linewidth=2,
                )

        plt.tight_layout()
        plt.savefig(f"stl_mapping/Planning/figures/polytopes/{robot_type}_{kind}_polytope.png")
        # plt.show()

    return A_inscribed, b_inscribed, radius_nominal, radius_eff, gyro_bound


if __name__ == "__main__":
    # --- BlueROV2 Analysis ---
    print("--- BlueROV2 Analysis ---")
    J_bluerov = approx_inertia_box(m=11.5, L=0.457, W=0.338, H=0.254)
    W_max_bluerov = 2.0

    A_F_br, b_F_br, rF_nom_br, rF_eff_br, _ = compute_polytope(
        "bluerov", "force", "icosahedron", True
    )
    print("Force polytope A matrix:\n", A_F_br)
    print("Force polytope b vector:\n", b_F_br)
    np.savez("stl_mapping/Planning/solutions/polytopes/force_polytope_bluerov.npz", A=A_F_br, b=b_F_br)

    A_T_br, b_T_br, rT_nom_br, rT_eff_br, gyro_br = compute_polytope(
        "bluerov", "torque", "cube", True, J=J_bluerov, W_max=W_max_bluerov
    )
    print("Torque polytope A matrix:\n", A_T_br)
    print("Torque polytope b vector:\n", b_T_br)
    np.savez("stl_mapping/Planning/solutions/polytopes/torque_polytope_bluerov.npz", A=A_T_br, b=b_T_br)

    # --- ATMOS Robot Analysis ---
    print("\n--- ATMOS Robot Analysis ---")
    A_F_atmos, b_F_atmos, rF_nom_atmos, rF_eff_atmos, _ = compute_polytope(
        "atmos", "force", "cube", True
    )
    print("Force polytope A matrix:\n", A_F_atmos)
    print("Force polytope b vector:\n", b_F_atmos)
    np.savez("stl_mapping/Planning/solutions/polytopes/force_polytope_atmos.npz", A=A_F_atmos, b=b_F_atmos)

    A_T_atmos, b_T_atmos, rT_nom_atmos, rT_eff_atmos, _ = compute_polytope(
        "atmos", "torque", "cube", True
    )
    print("Torque polytope A matrix:\n", A_T_atmos)
    print("Torque polytope b vector:\n", b_T_atmos)
    np.savez("stl_mapping/Planning/solutions/polytopes/torque_polytope_atmos.npz", A=A_T_atmos, b=b_T_atmos)

    # --- Astrobee Robot Analysis ---
    print("\n--- Astrobee Robot Analysis ---")
    # Using placeholder values for inertia and angular speed
    J_astrobee = np.diag([0.315, 0.315, 0.315])
    W_max_astrobee = 5.0

    A_F_ast, b_F_ast, rF_nom_ast, rF_eff_ast, _ = compute_polytope(
        "astrobee", "force", "cube", True
    )
    print("Force polytope A matrix:\n", A_F_ast)
    print("Force polytope b vector:\n", b_F_ast)
    np.savez("stl_mapping/Planning/solutions/polytopes/force_polytope_astrobee.npz", A=A_F_ast, b=b_F_ast)

    A_T_ast, b_T_ast, rT_nom_ast, rT_eff_ast, gyro_ast = compute_polytope(
        "astrobee", "torque", "icosahedron", True, J=J_astrobee, W_max=W_max_astrobee
    )
    print("Torque polytope A matrix:\n", A_T_ast)
    print("Torque polytope b vector:\n", b_T_ast)
    np.savez("stl_mapping/Planning/solutions/polytopes/torque_polytope_astrobee.npz", A=A_T_ast, b=b_T_ast)
