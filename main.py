import pygame as pg
import moderngl as mgl
import numpy as np
from pyrr import Matrix44

# --- SHADER SOURCE CODE ---
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;

uniform mat4 m_proj;
uniform mat4 m_view;

out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = m_proj * m_view * vec4(in_position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_color, 1.0);
}
"""

class VoxelRenderer:
    def __init__(self):
        # 1. Initialize Pygame and ModernGL
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        
        self.win_size = (800, 600)
        pg.display.set_mode(self.win_size, pg.OPENGL | pg.DOUBLEBUF)
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        
        self.ctx = mgl.create_context()
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
        
        # 2. Compile Shaders
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        
        # 3. Setup Camera Data
        self.camera_pos = np.array([16.0, 20.0, 50.0], dtype='float32')
        self.camera_pitch = 0.0
        self.camera_yaw = -90.0
        self.up = np.array([0.0, 1.0, 0.0], dtype='float32')
        self.forward = np.array([0.0, 0.0, -1.0], dtype='float32')
        
        # 4. Generate World Grid (32x32x32)
        self.world_size = 32
        self.voxels = np.zeros((self.world_size, self.world_size, self.world_size), dtype='uint8')
        
        # Procedural generation: fill lower half with "dirt/stone" blocks
        for x in range(self.world_size):
            for z in range(self.world_size):
                # Simple terrain height variation
                height = int(8 + 4 * np.sin(x * 0.2) * np.cos(z * 0.2))
                self.voxels[x, :height, z] = 1 

        # 5. Build and Bind Mesh
        self.vbo = None
        self.vao = None
        self.build_mesh()
        
        # Matrices
        proj = Matrix44.perspective_projection(60.0, self.win_size[0] / self.win_size[1], 0.1, 100.0)
        self.prog['m_proj'].write(proj.astype('float32').tobytes())
        
        self.clock = pg.time.Clock()

    def build_mesh(self):
        """Iterates through the voxel grid and builds vertices ONLY for exposed faces."""
        vertices = []
        
        # Local definition of 6 cube face offsets and matching vertex coordinates
        # Format: 4 vertices per face (X, Y, Z) + (R, G, B colors based on face direction)
        adj_offsets = [
            (0, 0, 1),  # Front
            (0, 0, -1), # Back
            (1, 0, 0),  # Right
            (-1, 0, 0), # Left
            (0, 1, 0),  # Top
            (0, -1, 0)  # Bottom
        ]
        
        # Loop through every grid coordinate
        for x in range(self.world_size):
            for y in range(self.world_size):
                for z in range(self.world_size):
                    if self.voxels[x, y, z] == 0:
                        continue
                    
                    # Check all 6 surrounding neighbors
                    for i, (dx, dy, dz) in enumerate(adj_offsets):
                        nx, ny, nz = x + dx, y + dy, z + dz
                        
                        # Hidden Face Culling: Render face if neighbor is air or out of bounds
                        if 0 <= nx < self.world_size and 0 <= ny < self.world_size and 0 <= nz < self.world_size:
                            if self.voxels[nx, ny, nz] != 0:
                                continue # Blocked, do not render this face
                        
                        # Append face geometry and color variations (shading effect)
                        c = 0.5 + 0.1 * i # Basic light variance per face orientation
                        color = [c * 0.2, c * 0.8, c * 0.3] # Greenish tint
                        
                        # Add raw polygon triangles for the visible face
                        self.append_face_triangles(vertices, x, y, z, i, color)
                        
        if not vertices:
            return
            
        vertex_data = np.array(vertices, dtype='float32')
        self.vbo = self.ctx.buffer(vertex_data.tobytes())
        # Position (3 floats), Color (3 floats)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f 3f', 'in_position', 'in_color')])

    def append_face_triangles(self, buffer, x, y, z, face_idx, col):
        # Local vertex coordinate definitions mapping relative to block position (x,y,z)
        v = [
            [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],
            [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]
        ]
        # Map indices into pairs of triangles (6 vertices total per square face)
        face_map = [
            [4, 5, 6, 4, 6, 7], # Front
            [1, 0, 3, 1, 3, 2], # Back
            [5, 1, 2, 5, 2, 6], # Right
            [0, 4, 7, 0, 7, 3], # Left
            [3, 2, 6, 3, 6, 7], # Top
            [0, 1, 5, 0, 5, 4]  # Bottom
        ]
        for vert_i in face_map[face_idx]:
            buffer.extend(v[vert_i])
            buffer.extend(col)

    def handle_input(self, dt):
        # Keyboard controls for flight navigation
        keys = pg.key.get_pressed()
        speed = 15.0 * dt
        if keys[pg.K_w]: self.camera_pos += self.forward * speed
        if keys[pg.K_s]: self.camera_pos -= self.forward * speed
        if keys[pg.K_a]: self.camera_pos -= np.cross(self.forward, self.up) * speed
        if keys[pg.K_d]: self.camera_pos += np.cross(self.forward, self.up) * speed
        
        # Mouse movement tracking for look directions
        rel_x, rel_y = pg.mouse.get_rel()
        self.camera_yaw += rel_x * 0.15
        self.camera_pitch -= rel_y * 0.15
        self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch))
        
        # Recalculate camera view target vectors
        yaw_rad = np.radians(self.camera_yaw)
        pitch_rad = np.radians(self.camera_pitch)
        self.forward[0] = np.cos(yaw_rad) * np.cos(pitch_rad)
        self.forward[1] = np.sin(pitch_rad)
        self.forward[2] = np.sin(yaw_rad) * np.cos(pitch_rad)
        self.forward = self.forward / np.linalg.norm(self.forward)

    def run(self):
        while True:
            dt = self.clock.tick(60) / 1000.0 # Delta time
            
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    pg.quit()
                    return
            
            self.handle_input(dt)
            
            # Render Pass
            self.ctx.clear(0.5, 0.8, 1.0) # Sky Blue Background
            
            # Construct View Matrix
            target = self.camera_pos + self.forward
            view = Matrix44.look_at(self.camera_pos, target, self.up)
            self.prog['m_view'].write(view.astype('float32').tobytes())
            
            if self.vao:
                self.vao.render()
                
            pg.display.flip()

if __name__ == '__main__':
    app = VoxelRenderer()
    app.run()
