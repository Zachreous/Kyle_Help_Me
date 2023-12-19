import numpy as np


def get_nodes():
    # Predefined node coordinates
    node_coords = [
        [0, 0, 0],
        [0, 0, 60],
        [0, 120, 60],
        [240, 0, 0],
        [240, 0, 60],
        [240, 120, 60],
        [120, 120, 60]
    ]
    return np.array(node_coords)

def get_elements():
    # Predefined elements
    elements = [
        [2, 3],
        [1, 3],
        [3, 6],
        [5, 6],
        [4, 6]
    ]
    #print(int(np.size(elements)/2)) #num of members
    return elements

def get_element_properties():
    # Predefined properties for each element
    properties = [
        [2500, 200, 108, 108, 160, 36],
        [2500, 200, 108, 108, 160, 36],
        [2500, 200, 216, 864, 592, 72],
        [2500, 200, 108, 108, 160, 36],
        [2500, 200, 108, 108, 160, 36]
    ]
    return properties

def get_forces():
    # Predefined forces
    forces = [
        ['Point Force', 3, 0, 0, -20],
        ['Point Force', 6, 0, 0, -20],
        ['Point Force', 7, 0, 0, -20]
    ]
    return forces

def create_global_force_vector():
    global_force_vector = [[0], [0], [-20], [0], [0], [0], [0], [0], [-20], [0], [0], [0]]
    return (np.array(global_force_vector))

#def create_global_force_vector(node_coords, forces):
    # Each node has 6 DOFs (3 translational, 3 rotational)
    #num_forces = len(forces)
    ##print(num_forces)
    #num_nodes = len(node_coords)
    #print(num_nodes)
    #global_force_vector = np.zeros(num_nodes * 6)
    #print(global_force_vector)

    # Map each force to its corresponding node and DOF
    #for force in forces:
        #force_type, node_idx, fx, fy, fz = force
        # Calculate the index in the global force vector
        #index_base = (node_idx) * 6
        #print(index_base)

        # Assign the force components to their respective positions
        #global_force_vector[index_base - 1] = fx  # x-direction force
        #global_force_vector[index_base] = fy  # z-direction force
        #global_force_vector[index_base + 1] = fz  # y-direction force
        #for

    #print(global_force_vector)
    #return global_force_vector


#def ff_global_force_vector(dof_array, forces, bcs_dict):
    # Total number of degrees of freedom
    #total_dof = np.max(dof_array)
    #print(total_dof)
    #global_forces_vector = np.zeros(total_dof)

    #for force in forces:
        #force_type, node_idx, fx, fy, fz = force
        #node_dofs = dof_array[node_idx - 1]

        # Check if the node is free (not fixed or pinned)
        #bc = bcs_dict.get(node_idx)
        #if bc != "Fixed Node" and bc != "Pinned Node":
            # Map each force component to its corresponding DOF
            #if node_dofs[0] > 0:  # x-direction
                #global_forces_vector[node_dofs[0] - 1] += fx
            #if node_dofs[1] > 0:  # y-direction
                #global_forces_vector[node_dofs[1] - 1] += fy
            #if node_dofs[2] > 0:  # z-direction
                #global_forces_vector[node_dofs[2] - 1] += fz

            # Additional rotational DOF for released ends
            #if "Member End Release" in bcs_dict.values():
                # Add an additional column for rotation
                #global_forces_vector = np.append(global_forces_vector, 0)

    #print(global_forces_vector)
    #return global_forces_vector


def get_boundary_conditions(nodes):
    # Predefined boundary conditions I'm tired of type them out : )
    bcs = [['Fixed Node', 1],
           ['Fixed Node', 2],
           ['Fixed Node', 4],
           ['Fixed Node', 5]
    ]

    releases = []

    return bcs, releases

# Get nodes, elements, and boundary conditions
nodes = get_nodes()
elements = get_elements()
boundary_conditions, releases = get_boundary_conditions(nodes)

# Create a dictionary for quick lookup of boundary conditions by node index
bcs_dict = {bc[1]: bc[0] for bc in boundary_conditions}

# Initialize the DOF array and a dictionary to track DOFs assigned to each node
dof_array = np.zeros((len(elements), 12), dtype=int)  # Initialize with zeros for all DOFs
node_dof_tracker = {}  # Dictionary to track DOFs assigned to each node
dof_counter = 1

# Creates Connectivity Matrix
def assign_dofs(node_index, dof_counter):
    if node_index in node_dof_tracker:
        return node_dof_tracker[node_index], dof_counter  # Return existing DOFs for shared nodes

    bc = bcs_dict.get(node_index, 'Free')
    if bc == "Fixed Node":
        dofs = np.zeros(6, dtype=int)  # All DOFs are fixed
    elif bc == "Pinned Node":
        dofs = np.array([0, 0, 0, 1, 1, 1], dtype=int)  # Only rotations are allowed
    else:  # 'Free' or not defined
        dofs = np.array([dof_counter + i if dof == 1 else 0 for i, dof in enumerate(np.ones(6, dtype=int))])
        dof_counter += np.sum(dofs > 0)  # Update the counter for free DOFs

    node_dof_tracker[node_index] = dofs
    #print(dofs)
    #print(dof_counter)
    return dofs, dof_counter

# Assigning DOFs for each node in each element
for element_idx, element in enumerate(elements):
    dof_i, dof_counter = assign_dofs(element[0], dof_counter)
    dof_j, dof_counter = assign_dofs(element[1], dof_counter)

    dof_array[element_idx, :6] = dof_i
    dof_array[element_idx, 6:] = dof_j

# Adjust DOFs for releases
for release in releases:
    member_index, end = int(releases[1]) -1 , release[2]

    if end == 'start':
        dof_array[member_index, 3:6] = dof_counter
        dof_counter += 1  # Increment only once for all three rotational DOFs
    elif end == 'end':
        dof_array[member_index, 9:12] = dof_counter
        dof_counter += 1

print("Final Connectivity Matrix:")
print(dof_array)


import matplotlib.pyplot as plt

#Creates 3D Plot for your viewing pleasure

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
                      force[3], force[2], force[4],

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
    ax.set_ylabel('Y (inches)')
    ax.set_zlabel('Z (inches)')
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
nodes = get_nodes()
#print(nodes)
elements = get_elements()
forces = get_forces()
bcs, releases = get_boundary_conditions(nodes)
array_of_nodes = draw_frame(nodes, elements, forces, bcs, releases) #calls for functions with array,
                                                                    #need mult paramtrs in parathensis

def calculate_element_angles(nodes, elements):
    angles = [0, 0, 90, 0, 0]

    return np.array(angles)


def calculate_element_lengths(elements, nodes):
    lengths = []
    for i, element in enumerate(elements):
        if any(idx < 1 or idx > len(nodes) for idx in element):
            print(f"Element {i + 1} has invalid indices: {element}. Skipping.")
            continue

        start_node_index = element[0] - 1
        end_node_index = element[1] - 1

        start_node = np.array(nodes[start_node_index])
        end_node = np.array(nodes[end_node_index])

        length = np.linalg.norm(end_node - start_node)
        lengths.append(length)

        # Bug fixing viewing window for lengths
        #print(f"Element {i + 1}: Length = {length}")

    return lengths


def create_transformation_matrix(nodes, angles, array_of_nodes):
    transformation_matrices = []
    nodes = get_nodes()

    for m, nodes in enumerate(array_of_nodes, 1):
        theta = angles[m-1]   # Get the corresponding angle for the element
        # Extract start and end node coordinates
        start_node_coord = nodes[0]
        end_node_coord = nodes[1]

        # Coordinate differences
        dx = end_node_coord[0] - start_node_coord[0]
        dy = end_node_coord[1] - start_node_coord[1]
        dz = end_node_coord[2] - start_node_coord[2]

        # Length of the element
        L = np.sqrt(dx**2 + dy**2 + dz**2)

        # Direction cosines
        cx = dx / L
        cy = dy / L
        cz = dz / L

        # cos and sin of angle (theta), assume it's provided or calculated previously
        cxz = np.sqrt(cx**2 + cz**2)
        cxy = np.sqrt(cx**2 + cy**2)
        c = np.cos(np.radians(theta))
        s = np.sin(np.radians(theta))

        # Transformation matrix T
        T = np.zeros((12, 12))

        # Define lambda based on cxz value
        if cxz == 0:
            lamb = np.array([
                [cx, cy, cz],
                [-cy / cxy, cx / cxy, 0],
                [-cx * cz / cxy, -cy * cz / cxy, cxy]
            ])
        # lambda for typical members
        else:
            lamb = np.array([
                [cx, cy, cz],
                [(-cx * cy * c -cz * s) / cxz, cxz * c, (-cy * cz * c + cx * s)
                 / cxz],
                [(cx * cy * s -cz * c) / cxz, -cxz * s, (cy * cz * s + cx * c)
                 / cxz]
            ])
        # Populate the transformation matrix T with lambda values
        for k in range(4):
            for i in range(3):
                T[0 + 3 * k, i + 3 * k] = lamb[0, i]
                T[1 + 3 * k, i + 3 * k] = lamb[1, i]
                T[2 + 3 * k, i + 3 * k] = lamb[2, i]

        T = np.transpose(T)
        transformation_matrices.append(T)
        #print(f"Transformation Matrix for Element {m}:")
        #print(T)

    return transformation_matrices


# Call of all prior functions
nodes = get_nodes()
elements = get_elements()
properties = get_element_properties()
forces = get_forces()
theta = calculate_element_angles(nodes, elements)
angles = calculate_element_angles(nodes, elements)
element_lengths = calculate_element_lengths(elements, nodes)
create_transformation_matrix(nodes, angles, array_of_nodes)
lengths = calculate_element_lengths(elements, nodes)

def frame_element_stiffness(element, elements, nodes, properties, transformation_matrices, lengths):

    k_global_matrices = []

    for index in range(len(elements)):
        # Extract element length
        L = lengths[index]
        #L = np.linalg.norm(np.array(nodes[element[1] - 1]) - np.array(nodes[element[0] - 1]))
        #print(L)
        #print(index)
        # Extract properties for this element
        E, G, Iy, Iz, J, A = properties[index]
        #print(A*E/L)
        #print(E)
        #print(Iy)
        #print(Iz)
        #print(J)
        #print(A)
        k_local = np.array([
            [A * E / L, 0, 0, 0, 0, 0, -A * E / L, 0, 0, 0, 0, 0],
            [0, (12 * E * Iz) / L ** 3, 0, 0, 0, (6 * E * Iz) / L ** 2, 0, (-12 * E * Iz) / L ** 3, 0, 0, 0,
             (6 * E * Iz) / L ** 2],
            [0, 0, (12 * E * Iy) / L ** 3, 0, (-6 * E * Iy) / L ** 2, 0, 0, 0, (-12 * E * Iy) / L ** 3, 0,
             (-6 * E * Iy) / L ** 2, 0],
            [0, 0, 0, (G * J) / L, 0, 0, 0, 0, 0, (-G * J) / L, 0, 0],
            [0, 0, (-6 * E * Iy) / L ** 2, 0, (4 * E * Iy) / L, 0, 0, 0, (6 * E * Iy) / L ** 2, 0, (2 * E * Iy) / L,
            0],
            [0, (6 * E * Iz) / L ** 2, 0, 0, 0, (4 * E * Iz) / L, 0, (-6 * E * Iz) / L ** 2, 0, 0, 0, (2 * E * Iz) / L],
            [(-A * E) / L, 0, 0, 0, 0, 0, (A * E / L), 0, 0, 0, 0, 0],
            [0, (-12 * E * Iz) / L ** 3, 0, 0, 0, (-6 * E * Iz) / L ** 2, 0, (12 * E * Iz) / L ** 3, 0, 0, 0,
             (-6 * E * Iz) / L ** 2],
            [0, 0, (-12 * E * Iy) / L ** 3, 0, (6 * E * Iy) / L ** 2, 0, 0, 0, (12 * E * Iy) / L ** 3, 0,
             (6 * E * Iy) / L ** 2, 0],
            [0, 0, 0, (-G * J) / L, 0, 0, 0, 0, 0, (G * J) / L, 0, 0],
            [0, 0, (-6 * E * Iy) / L ** 2, 0, (2 * E * Iy) / L, 0, 0, 0, (6 * E * Iy) / L ** 2, 0, (4 * E * Iy) / L,
            0],
            [0, (6 * E * Iz) / L ** 2, 0, 0, 0, (2 * E * Iz) / L, 0, (-6 * E * Iz) / L ** 2, 0, 0, 0, (4 * E * Iz) / L]
        ])

        # Calculate the transformation matrix for this element
        T = transformation_matrices[index]

        # Rotate the local stiffness matrix to global coordinates
        k_global = T @ k_local @ np.linalg.inv(T)
        #print(np.linalg.inv(T))
        #Debugging k global stiffness matrices for each element
        #print(f"Local Stiffness Matrix for Element {index + 1}:")
        #print(k_local)
        #print(f"Global Stiffness Matrix for Element {index + 1}:")
        #print(k_global)

        # Store stiffness matrix for each element
        k_global_matrices.append(k_global)

    return k_global_matrices

# Example usage
nodes = get_nodes()
elements = get_elements()
thetas = calculate_element_angles(nodes, elements)  # Calculate or provide theta for each element
transformation_matrices = create_transformation_matrix(nodes, angles, array_of_nodes)
k_global_matrices = frame_element_stiffness(element, elements, nodes, properties, transformation_matrices, lengths)

def Kstr(elements, dof_array, k_global_matrices):
    """
    Assemble the global structure stiffness matrix.

    Parameters:
    - elements: List of elements
    - properties: List of material and geometric properties for each element
    - dof_array: Array of degrees of freedom for each element

    Returns:
    - Global structure stiffness matrix
    """

    # Total number of degrees of freedom
    total_dof = max(map(max, dof_array))
    #print(f"Total DOF: {total_dof}")

    # Initialize the global stiffness matrix
    K_structure = np.zeros((total_dof, total_dof))

    # Assemble the structure stiffness matrix
    for m in range(len(elements)):
        #print(len(elements))
        k_local = k_global_matrices[m]

        # Get the degrees of freedom for this element
        element_dof = dof_array
        #print(element_dof)

        # Add the contribution of the local stiffness matrix to the global matrix
        for j in range(total_dof):
            for i in range(total_dof):

                p = dof_array[m, i] - 1
                q = dof_array[m, j] - 1

                if p >= 0 and q >= 0:  # Check to ensure indices are within bounds
                    K_structure[p, q] += k_local[i, j]
    if np.linalg.det(K_structure) == 0:
        print('Structure is unstable. Please Check the boundary conditions of the structure.')
    #print(K_structure)
    return K_structure


# Degbugging output, print inside function is being directed elsewhere
#try:
    # Assuming dof_array and k_global_matrices are already defined
    #K_structure = Kstr(elements, dof_array, k_global_matrices)
#except Exception as e:
    #print(f"An error occurred: {e}")

# Get the local stiffness matrix for this element
elements = get_elements()
bcs = get_boundary_conditions(nodes) #calling boundary conditions for support_conditions
nodes = get_nodes()  # Assuming you have a function to get the nodes

# Example usage
nodes = get_nodes()  # Assuming get_nodes() returns a list of node indices
boundary_conditions = get_boundary_conditions(nodes)  # Get boundary conditions from the user


def member_fixed_end_forces(elements, forces, node_coords):
    """
    Calculate the global fixed end forces for each member in the structure.

    Parameters:
    - elements: List of lists, each sublist contains the indices of the start and end node of each element.
    - forces: List of forces applied to each element, including the node index of force application and force components.
    - node_coordinates: Numpy array containing coordinates for each node.

    Returns:
    - A list of numpy arrays, each representing the global fixed end forces for each element.
    """
    fixed_end_forces_list = []

    # Determine the support condition for each node
    node_support_conditions = {node_idx: 'fixed' for node_idx in range(1, len(node_coords) + 1)}
    for force in forces:
        if force[0] in ['Fixed Node', 'Pinned Node', 'Member End Release']:
            node_idx = force[0]
            condition = force[0].lower().split()[0]  # 'fixed', 'pinned', or 'released'
            node_support_conditions[node_idx] = condition

    for element in elements:
        start_node_idx, end_node_idx = element
        #print(start_node_idx)
        #print(end_node_idx)
        start_support = node_support_conditions[start_node_idx]
        #print(start_support)
        end_support = node_support_conditions[end_node_idx]
        #print(end_support)
        start_node_coord = node_coords[start_node_idx - 1]
        #print(start_node_coord)
        end_node_coord = node_coords[end_node_idx - 1]
        #print(end_node_coord)

        # Initialize fixed end forces for the element
        fixed_end_forces = np.zeros(12).T
        #print(fixed_end_forces)

        force_type, force_node_idx, fx, fz, fy = force
        #print(force_type)
        #print(force_node_idx)
        #print(fx)
        #print(fy)
        #print(fz)
        if force_type == 'Point Force':
            node_coords = get_nodes()  # Assuming this returns your node coordinates
            length = np.linalg.norm(end_node_coord - start_node_coord)  # Calculate the length of the element

            if start_support == 'fixed' and end_support == 'fixed':
                # Fixed-Fixed condition formulas for point load
                fixed_end_forces = [[-fx / 2], [-fy / 2], [-fz / 2], [fx * length / 8], [fy * length / 8], [fz * length / 8],
                                [-fx / 2], [-fy / 2], [-fz / 2], [-fx * length / 8], [-fy * length / 8], [-fz * length / 8]
                                ]
                #print(fixed_end_forces)
            elif start_support == 'fixed' and end_support == 'pinned':
                # Fixed-Pinned condition formulas for point load
                fixed_end_forces = [-fx / 2, -fz / 2, -fy / 2, 3 * -fx * length / 16, 3 * -fz * length / 16,
                                    3 * -fy * length / 16, fx / 2, fz / 2, fy / 2, 0, 0, 0
                                    ]

            elif start_support == 'released' and end_support == 'released':
                # Pinned-Pinned condition formulas for point load
                fixed_end_forces = [-fx / 2, -fz / 2, -fy / 2, 0, 0, 0, fx / 2, fz / 2, fy / 2, 0, 0, 0
                                    ]

        elif force_type == 'Distributed Force':
            # Extract distributed load information
            start_node, end_node, fx, fz, fy = load
            length = np.linalg.norm(end_node_coord - start_node_coord)  # Calculate the length of the element

            if start_support == 'fixed' and end_support == 'fixed':
                # Fixed-Fixed condition formulas for UDL
                fixed_end_forces = [fx * length / 2, fz * length / 2, fy * length / 2,  fx * length ** 2 / 12,
                                    fz * length ** 2 / 12, -fy * length ** 2 / 12, fx * length / 2, fz * length / 2,
                                    fy * length / 2, fx * length ** 2 / 12, fz * length ** 2 / 12,
                                    fy * length ** 2 / 12
                                    ]

            elif start_support == 'fixed' and end_support == 'pinned':

                # Fixed-Pinned condition formulas for UDL
                fixed_end_forces = [-5 * fx * length / 8, -5 * fz * length / 8, -5 * fy * length / 8,
                                    -fx * length ** 2 / 8, -fz * length ** 2 / 8, -fy * length ** 2 / 8,
                                    3 * fx * length / 8, 3 * fz * length / 8, 3 * fy * length / 8, 0, 0, 0
                                    ]

            elif start_support == 'released' and end_support == 'released':
                # Pinned-Pinned condition formulas for UDL
                fixed_end_forces = [-fx * length / 2, -fz * length / 2, -fy * length / 2, 0, 0, 0,
                                    fx * length / 2, fz * length / 2, fy * length / 2, 0, 0, 0
                                    ]
            #print(np.array(fixed_end_forces_list))
        # Add the calculated fixed end forces to the list
        fixed_end_forces_list.append(fixed_end_forces)
    #print(np.array(fixed_end_forces_list))
    return np.array(fixed_end_forces_list)




node_coords = get_nodes()
fixed_end_forces_list = member_fixed_end_forces(elements, forces, node_coords)
#print(fixed_end_forces_list[0])
#print(fixed_end_forces_list[1])
#print(fixed_end_forces_list[2])
#print(fixed_end_forces_list[3])
#print(fixed_end_forces_list[4])

#Determine the memeber with unconstrained ends
def get_free_node_indexes(elements):
    """
    Get free node indexes from the user input.

    Returns:
    - A list of integers representing specific node indexes based on a defined condition.
    """
    free_node_indexes = list(map(int, input("Enter the indexes of free nodes, separated by spaces: ").split()))
    matched_indexes = []

    for free_node_index in free_node_indexes:
        for element_row in elements:
            if free_node_index in element_row and free_node_index not in matched_indexes:
                matched_indexes.append(free_node_index)

    return matched_indexes

#creating fixed end forces based on releases and the number of members with free dof on both ends
def global_fixed_end_forces(elements, forces, dof_array, node_coords, fixed_end_forces_list):
    free_node_indexes = get_free_node_indexes(elements)
    global_fefs = []

    for free_node_index in free_node_indexes:
        if free_node_index - 1 < len(fixed_end_forces_list):
            global_fef = fixed_end_forces_list[free_node_index - 1]
            global_fefs.append(global_fef)
            #print(len(global_fef))
    #print(dof_counter)
    dof_current = dof_counter - 1
    while dof_current < dof_current:
        new_dof = [0, 0, 0]
        global_fefs.append(new_dof)
        dof_current += 1
    return np.array(global_fefs)


node_coords = get_nodes()

def calculate_global_displacements(K_structure, global_force_vector, global_fefs):
    """
    Calculate the global displacements in the structure.

    Parameters:
    - K_structure: Global stiffness matrix of the structure.
    - applied_forces: Vector of applied forces at the free degrees of freedom.
    - global_fefs: Global fixed end forces vector.

    Returns:
    - Global displacements vector.
    """
    # Check if K_structure is invertible
    if np.linalg.det(K_structure) == 0:
        raise ValueError("The stiffness matrix is singular and cannot be inverted.")

    # Calculate the difference between applied forces and fixed end forces
    force_diff = np.array(global_force_vector) - global_fefs
    #print(global_force_vector)
    #print(force_diff)
    #print(np.array(global_force_vector))
    #print(global_fefs)

    # Calculate global displacements
    global_displacements = np.linalg.inv(K_structure).dot(force_diff)

    print(global_displacements)
    return global_displacements


K_structure = Kstr(elements, dof_array, k_global_matrices)
#print(K_structure)
#print(np.linalg.inv(K_structure))
global_force_vector = create_global_force_vector()#dof_array, forces)
#print(global_force_vector)
global_fefs = global_fixed_end_forces(elements, forces, dof_array, node_coords, fixed_end_forces_list)
global_displacements = calculate_global_displacements(K_structure, global_force_vector, global_fefs)



def member_end_displacements(global_displacements, connectivity_matrix, n_members):
    """
    Apply global displacements to the ends of each member.

    Parameters:
    - global_displacements: The global displacement vector for the entire structure.
    - connectivity_matrix: The connectivity matrix mapping global DoFs to member local DoFs.
    - n_members: The total number of members in the structure.

    Returns:
    - A list of 12-element arrays, each representing the displacements at the member ends.
    """
    # Initialize a list to hold displacement arrays for each member
    member_displacements = []

    for m in range(n_members):
        # Initialize the displacement array for this member
        member_disp = np.zeros(12)

        # Iterate through each of the 12 local DoFs for the member
        for i in range(12):
            # Get the global DoF corresponding to the local DoF
            global_dof = connectivity_matrix[m, i]

            # If the global DoF is not fixed (indicated by p not equal to 0)
            if global_dof != 0:
                # Assign the displacement from the global displacements vector
                member_disp[i] = global_displacements[global_dof - 1]  # Adjust for zero-based indexing

        member_displacements.append(member_disp)

    return member_displacements

K_structure = Kstr(elements, dof_array, k_global_matrices)
n_elements = get_elements()
forces = get_forces()


def member_internal_forces(member_length, point_loads, distributed_loads, end_forces):
    """
    Calculate and plot internal shear forces and bending moments within a member considering 3D forces.

    Parameters:
    - member_length: Length of the member.
    - point_loads: List of tuples representing point loads (magnitude in kips, position from start of member, direction as 'x', 'y', or 'z').
    - distributed_loads: List of tuples representing uniform distributed loads (magnitude in kips per inch, start position, end position, direction as 'x', 'y', or 'z').
    - end_forces: Tuple representing end shear and moments (start shear in kips, start moment in kip-inches, end shear in kips, end moment in kip-inches) for y and z directions.

    Note: This function assumes linear variation of shear due to point loads and parabolic variation of moment due to distributed loads.
    """

    # Initialize arrays to hold shear and moment values
    x_vals = np.linspace(0, member_length, 100)  # Positions along member for plotting
    # Initialize for y and z directions
    V_y = np.zeros_like(x_vals)
    M_y = np.zeros_like(x_vals)
    V_z = np.zeros_like(x_vals)
    M_z = np.zeros_like(x_vals)

    # Apply end conditions for y and z directions
    V_start_y, M_start_y, V_end_y, M_end_y, V_start_z, M_start_z, V_end_z, M_end_z = end_forces
    V_y[0] += V_start_y
    M_y[0] += M_start_y
    V_z[0] += V_start_z
    M_z[0] += M_start_z

    # Process point loads for y and z directions
    for P, a, direction in point_loads:
        if direction == 'y':
            load_idx = np.searchsorted(x_vals, a)
            V_y[load_idx:] += P
            M_y[load_idx:] += P * (x_vals[load_idx:] - a)
        elif direction == 'z':
            load_idx = np.searchsorted(x_vals, a)
            V_z[load_idx:] += P
            M_z[load_idx:] += P * (x_vals[load_idx:] - a)

    # Process distributed loads for y and z directions
    for w, a, b, direction in distributed_loads:
        idx_start = np.searchsorted(x_vals, a)
        idx_end = np.searchsorted(x_vals, b)
        if direction == 'y':
            for idx in range(idx_start, idx_end):
                V_y[idx] += w * (x_vals[idx] - a)
                M_y[idx] += 0.5 * w * (x_vals[idx] - a)**2
        elif direction == 'z':
            for idx in range(idx_start, idx_end):
                V_z[idx] += w * (x_vals[idx] - a)
                M_z[idx] += 0.5 * w * (x_vals[idx] - a)**2

    # Apply end shear and moment for y and z directions
    V_y[-1] += V_end_y
    M_y[-1] += M_end_y
    V_z[-1] += V_end_z
    M_z[-1] += M_end_z

    # Plotting the shear force diagram (SFD) and bending moment diagram (BMD)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create subplots for y and z directions

    # Plot Shear Force Diagram (SFD) for y direction
    axs[0, 0].plot(x_vals, V_y, label='Shear Force V_y (kips)')
    axs[0, 0].fill_between(x_vals, V_y, alpha=0.3)
    axs[0, 0].set_title('Shear Force Diagram V_y (SFD)')
    axs[0, 0].set_ylabel('Shear Force V_y (kips)')
    axs[0, 0].grid(True)

    # Plot Bending Moment Diagram (BMD) for y direction
    axs[1, 0].plot(x_vals, M_y, label='Bending Moment M_y (kip-in)')
    axs[1, 0].fill_between(x_vals, M_y, alpha=0.3)
    axs[1, 0].set_title('Bending Moment Diagram M_y (BMD)')
    axs[1, 0].set_xlabel('Length along member (inches)')
    axs[1, 0].set_ylabel('Bending Moment M_y (kip-in)')
    axs[1, 0].grid(True)

    # Plot Shear Force Diagram (SFD) for z direction
    axs[0, 1].plot(x_vals, V_z, label='Shear Force V_z (kips)')
    axs[0, 1].fill_between(x_vals, V_z, alpha=0.3)
    axs[0, 1].set_title('Shear Force Diagram V_z (SFD)')
    axs[0, 1].set_ylabel('Shear Force V_z (kips)')
    axs[0, 1].grid(True)

    # Plot Bending Moment Diagram (BMD) for z direction
    axs[1, 1].plot(x_vals, M_z, label='Bending Moment M_z (kip-in)')
    axs[1, 1].fill_between(x_vals, M_z, alpha=0.3)
    axs[1, 1].set_title('Bending Moment Diagram M_z (BMD)')
    axs[1, 1].set_xlabel('Length along member (inches)')
    axs[1, 1].set_ylabel('Bending Moment M_z (kip-in)')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    return V_y, M_y, V_z, M_z




