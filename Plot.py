import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_nodes():
    num_nodes = int(input("Enter the number of nodes: "))
    node_coords = np.zeros((num_nodes, 3))  # Initialize an array with 3 columns (x, y, z)

    for i in range(num_nodes):
        x, z, y = map(float, input(f"Enter coordinates for node {i + 1} (x z y): ").split())
        node_coords[i, :] = [x, y, z]

    return node_coords, num_nodes


def get_elements():
    num_elements = int(input("Enter the number of elements: "))
    elements = []
    for i in range(num_elements):
        start_node, end_node = map(int, input(f"Enter nodes for element {i + 1} (start_node end_node): ").split())
        elements.append([start_node, end_node])
    return num_elements, elements


def get_element_with_properties(num_elements):
    properties = []
    for i in range(num_elements):
        print("Enter properties for element: ", i + 1)
        E = float(input(f"Enter Young's modulus (E) for element {i+1}: "))
        G = float(input(f"Enter shear modulus (G) for element {i+1}: "))
        Iy = float(input(f"Enter moment of inertia about the y-axis (Iy) for element {i+1}: "))
        Iz = float(input(f"Enter moment of inertia about the z-axis (Iz) for element {i+1}: "))
        J = float(input(f"Enter torsion coefficient (J) for element {i+1}: "))
        A = float(input(f"Enter cross-sectional area (A) for element {i+1}: "))
        properties.append([E, G, Iy, Iz, J, A])
    return properties


def get_forces():
    forces = []
    while True:
        print("Choose the type of force:")
        print("1. Point Force")
        print("2. Distributed Force")
        print("3. Done")
        choice = int(input("Enter your choice (1-3): "))

        if choice == 1:
            node = int(input("Enter the node index for the point force: "))
            force_x, force_y, force_z = map(float, input("Enter force components (x y z): ").split())
            forces.append(['Point Force', node, force_x, force_y, force_z])
        elif choice == 2:
            start_node = int(input("Enter the start node index for the distributed force: "))
            end_node = int(input("Enter the end node index for the distributed force: "))
            force_x, force_y, force_z = map(float, input("Enter force components (x y z): ").split())
            forces.append(['Distributed Force', start_node, end_node, force_x, force_y, force_z])
        elif choice == 3:
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

    return forces


def get_boundary_conditions():
    bcs = []
    releases = []
    while True:
        print("Choose the type of boundary condition and member end releases:")
        print("1. Fixed Node")
        print("2. Pinned Node")
        print("3. Member End Release")
        print("4. Done")
        choice = int(input("Enter your choice (1-4): "))

        if choice == 1:
            node = int(input("Enter the node index for the fixed node: "))
            bcs.append(['Fixed Node', node])
        elif choice == 2:
            node = int(input("Enter the node index for the pinned node: "))
            bcs.append(['Pinned Node', node])
        elif choice == 3:
            element = int(input("Enter the element number for the release: "))
            end = input("Enter which end of the member has the release (start/end): ")
            releases.append(['Member End Release', element, end])
        elif choice == 4:
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

    return bcs, releases


def calculate_element_angles(nodes, elements):
    angles = []
    for i, element in enumerate(elements):
        start_node = np.array(nodes[element[0] - 1])
        end_node = np.array(nodes[element[1] - 1])
        #Bug fixing viewing window for angles
        #print(f"Node {i + 1}: i end = {start_node}")
        #print(type(start_node))
        #print(f"Node {i + 1}: j end = {end_node}")
        #print(type(end_node))
        # Calculate the direction vector of the element
        direction_vector = end_node - start_node

        # Calculate the angle from the z-axis
        angle_z = np.degrees(np.arccos(direction_vector[1] / np.linalg.norm(direction_vector)))

        angles.append(angle_z)

        #print(f"Element {i + 1}: Angle from Z = {angle_z} degrees")
        #print(angles)

    return np.array(angles)

def calculate_element_lengths(elements, nodes):
    lengths = []
    for element in elements:
        start_node = np.array(nodes[element[0] - 1])
        end_node = np.array(nodes[element[1] - 1])
        length = np.linalg.norm(end_node - start_node)
        lengths.append(length)
    return lengths

#Creates 3D Plot for your viewing pleasure

import matplotlib.pyplot as plt

def draw_frame(nodes, elements, forces, bcs, releases):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes
    for i, node in enumerate(nodes):
        ax.scatter(node[0], node[2], node[1], c='b', marker='o', label=f'Node {i + 1}')
        ax.text(node[0], node[2], node[1], f'{i + 1}', fontsize=8, color='black')

    # Plot elements
    for element in elements:
        start_node = nodes[element[0] - 1]
        end_node = nodes[element[1] - 1]
        ax.plot([start_node[0], end_node[0]],
                [start_node[2], end_node[2]],
                [start_node[1], end_node[1]], c='k')

    # Plot releases as circles at member ends
    for release in releases:
        member_index, end = release[1], release[2]
        # Check if the release is at the 'start' or 'end' of the member
        if end == 'start':
            # Use the first node index from the elements list for this member
            node_index = elements[member_index - 1][0] - 1
        else:
            # Use the second node index from the elements list for this member
            node_index = elements[member_index - 1][1] - 1

        # Retrieve the correct node coordinates using the node index
        node = nodes[node_index]

        # Plot the release circle at the correct node position
        ax.scatter(node[0], node[2], node[1], c='r', marker='o', s=25, label=f'Release at Node {node_index + 1}')

    # Plot forces
    for force in forces:
        if force[0] == 'Point Force':
            node = nodes[force[1] - 1]
            ax.quiver(node[0], node[2], node[1],
                      force[2], force[3], force[4],

                      #Creates colored arrow
                      color='r', label=f'Point Force at Node {force[1]}', arrow_length_ratio=0.1)

        elif force[0] == 'Distributed Force':
            start_node = nodes[force[1] - 1]
            end_node = nodes[force[2] - 1]
            force_vector = np.array([force[3], force[4], force[5]])

            # Calculate the midpoint to give the position of the resultant force
            mid_point = (np.array(start_node) + np.array(end_node)) / 2
            ax.quiver(mid_point[0], mid_point[2], mid_point[1],
                      force_vector[0], force_vector[2], force_vector[1],
                      color='b', label=f'Distributed Force from Node {force[1]} to Node {force[2]}',
                      arrow_length_ratio=0.1)

        # Plot boundary conditions
        for bc in boundary_conditions:
            node = nodes[bc[1] - 1]
            if bc[0] == 'Fixed Node':
                ax.scatter(node[0], node[2], node[1], c='g', marker='s', s=100,
                           label='Fixed Node' if bc[1] == 1 else "")
            elif bc[0] == 'Pinned Node':
                ax.scatter(node[0], node[2], node[1], c='orange', marker='^', s=100,
                           label='Pinned Node' if bc[1] == 1 else "")

    # Set labels
    ax.set_xlabel('X (inches)')
    ax.set_ylabel('Z (inches)')  # Updated for Y as depth
    ax.set_zlabel('Y (inches)')  # Updated for Z as vertical
    ax.set_title('3D Frame Finite Element Analysis')

#User Interface quality of life adjustements, for your viewing pleasure

    # Flip the Z and Y axis
    ax.set_xlim(ax.get_xlim()[::1])
    ax.set_ylim(ax.get_ylim()[::1])  # Ensure Y-axis is oriented correctly
    ax.set_zlim(ax.get_zlim()[::1])  # Ensure Z-axis is oriented correctly


    #Set_position([left, bottom, width, height]) of plot
    ax.set_position([0.1, 0.1, 0.4, 0.8])

    # Add text inside the plot
    ax.text2D(0.05, 1.2, "PLEASE CLOSE THE FIGURE TO CONTINUE ANALYSIS", transform=ax.transAxes, fontsize=12,
              color='red')

    # Console notice before showing the plot
    print("Please close the figure window to continue analysis.")

    # Move legend to the side, Otherwise the legend centers on the graph
    ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0.)

    plt.show()


    # Create an array of nodes from start and end nodes
    array_of_nodes = []
    for element in elements:
        start_node = nodes[element[0] - 1]
        end_node = nodes[element[1] - 1]
        array_of_nodes.append([start_node, end_node])

    return array_of_nodes


#Call node array from draw frame function:
node_coords = get_nodes()
elements = get_elements()
forces = get_forces()
bcs, releases = get_boundary_conditions()
array_of_nodes = draw_frame(elements, forces, bcs, releases) #calls for functions with array,
                                                                    #need mult paramtrs in parathensis
boundary_conditions, releases = get_boundary_conditions()
