import math
import random
from dataclasses import dataclass, field

import numpy as np
import moderngl
import pygame
from pygame.locals import DOUBLEBUF, OPENGL

import traceback
import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

ACversion = "0.6.1"

# ============================================================================
#  Config
# ============================================================================
WIDTH, HEIGHT = 1280, 720
WORLD_X, WORLD_Z = 64, 64
WORLD_Y = 32
CHUNK_SIZE = 16
WATER_LEVEL = 8  # Adjusted water level for flatter terrain

FOV_Y = 70.0
NEAR, FAR = 0.05, 300.0
MOUSE_SENS = 0.0025

WALK_SPEED = 6.0
GRAVITY = -24.0
JUMP_SPEED = 8.5
PLAYER_WIDTH = 0.6
PLAYER_HEIGHT = 1.8
EYE_HEIGHT = 1.6
COLLISION_EPS = 1e-4

# Block IDs
BLOCK_AIR     = 0
BLOCK_GRASS   = 1
BLOCK_DIRT    = 2
BLOCK_STONE   = 3
BLOCK_WOOD    = 4
BLOCK_LEAVES  = 5
BLOCK_SAND    = 6
BLOCK_GRAVEL  = 7
BLOCK_BEDROCK = 8
BLOCK_PLANKS  = 9
BLOCK_AGOUTI  = 10
BLOCK_WATER   = 11

BLOCK_BREAK_TIMES = {
    BLOCK_LEAVES:  0.2,
    BLOCK_DIRT:    0.5,
    BLOCK_GRASS:   0.5,
    BLOCK_SAND:    0.5,
    BLOCK_GRAVEL:  0.6,
    BLOCK_PLANKS:  0.8,
    BLOCK_WOOD:    1.2,
    BLOCK_STONE:   1.8,
    BLOCK_BEDROCK: float('inf'),
    BLOCK_AGOUTI:  0.5,
    BLOCK_WATER:   float('inf'),
}

# Mapping block IDs to texture files
TEXTURE_FILES = {
    BLOCK_GRASS:   "textures/grass.jpg",
    BLOCK_DIRT:    "textures/dirt.jpg",
    BLOCK_STONE:   "textures/stone.webp",
    BLOCK_WOOD:    "textures/log.webp",
    BLOCK_LEAVES:  "textures/leaf.jpg",
    BLOCK_SAND:    "textures/sand.png",
    BLOCK_GRAVEL:  "textures/stone.webp",
    BLOCK_BEDROCK: "textures/bedrock.jpeg",
    BLOCK_PLANKS:  "textures/plank.jpeg",
    BLOCK_AGOUTI:  "textures/brownagouti.jpg",
    BLOCK_WATER:   "textures/water.jpeg",
}

# Face Direction Definitions
FACES = [
    ((0, 1, 0), 1.00, [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]),
    ((0, -1, 0), 0.40, [(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)]),
    ((0, 0, -1), 0.70, [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]),
    ((0, 0, 1), 0.70, [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]),
    ((-1, 0, 0), 0.60, [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]),
    ((1, 0, 0), 0.60, [(1, 0, 1), (1, 1, 1), (1, 1, 0), (1, 0, 0)]),
]

UV_QUAD = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

VERT_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in vec2 in_uv;
in float in_brightness;
in float in_tile_idx;

out vec2 v_uv;
out float v_brightness;
out float v_tile_idx;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_uv = in_uv;
    v_brightness = in_brightness;
    v_tile_idx = in_tile_idx;
}
"""

FRAG_SHADER = """
#version 330
uniform sampler2DArray atlas;

in vec2 v_uv;
in float v_brightness;
in float v_tile_idx;

out vec4 f_color;

void main() {
    vec4 tex_color = texture(atlas, vec3(v_uv, v_tile_idx));
    // Apply slight transparency if drawing water
    if (v_tile_idx == 10.0) { // Water tile index
        tex_color.a = 0.65;
    }
    f_color = vec4(tex_color.rgb * v_brightness, tex_color.a);
}
"""

OVERLAY_VERT_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in vec2 in_uv;
in float in_tile_idx;

out vec2 v_uv;
out float v_tile_idx;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_uv = in_uv;
    v_tile_idx = in_tile_idx;
}
"""

OVERLAY_FRAG_SHADER = """
#version 330
uniform sampler2DArray atlas;
uniform float dark_factor;

in vec2 v_uv;
in float v_tile_idx;

out vec4 f_color;

void main() {
    vec4 tex_color = texture(atlas, vec3(v_uv, v_tile_idx));
    f_color = vec4(tex_color.rgb * (1.0 - dark_factor * 0.85), tex_color.a);
}
"""

LINE_VERT_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_position;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

LINE_FRAG_SHADER = """
#version 330
out vec4 f_color;

void main() {
    f_color = vec4(1.0, 1.0, 0.0, 1.0); // RGBA: Bright Yellow
}
"""

# ============================================================================
#  Texture Atlas Builder
# ============================================================================
class TextureAtlas:
    def __init__(self, ctx):
        self.ctx = ctx
        self.block_to_layer = {}
        self.texture_array = None
        self.load_textures()

    def load_textures(self):
        size = 64
        unique_blocks = list(TEXTURE_FILES.keys())
        num_layers = len(unique_blocks)

        atlas_data = bytearray()

        for idx, block_id in enumerate(unique_blocks):
            filename = TEXTURE_FILES[block_id]
            self.block_to_layer[block_id] = idx

            try:
                img_path = resource_path(filename)
                img = pygame.image.load(img_path).convert_alpha()
                img = pygame.transform.scale(img, (size, size))
            except Exception as e:
                print(f"Failed to load texture {filename}: {e}")
                img = pygame.Surface((size, size))
                img.fill((200, 0, 200))

            raw_data = pygame.image.tostring(img, "RGBA", False)
            atlas_data.extend(raw_data)

        # Create ModernGL Texture Array from bundled texture data
        self.texture_array = self.ctx.texture_array(
            (size, size, num_layers), 
            4, 
            bytes(atlas_data)
        )
        self.texture_array.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def get_layer(self, block_id):
        return float(self.block_to_layer.get(block_id, 0))

# ============================================================================
#  World Generation
# ============================================================================
def value_noise_2d(width, height, scale, seed):
    rng = random.Random(seed)
    gw, gh = width // scale + 2, height // scale + 2
    grid = [[rng.random() for _ in range(gh)] for _ in range(gw)]

    def smooth(t):
        return t * t * (3 - 2 * t)

    out = np.zeros((width, height), dtype=np.float32)
    for x in range(width):
        gx = x / scale
        x0 = int(gx)
        tx = smooth(gx - x0)
        for z in range(height):
            gz = z / scale
            z0 = int(gz)
            tz = smooth(gz - z0)
            v00, v10 = grid[x0][z0], grid[x0 + 1][z0]
            v01, v11 = grid[x0][z0 + 1], grid[x0 + 1][z0 + 1]
            a = v00 * (1 - tx) + v10 * tx
            b = v01 * (1 - tx) + v11 * tx
            out[x, z] = a * (1 - tz) + b * tz
    return out


def add_tree(blocks, tx, ty, tz):
    trunk_height = random.randint(4, 5)
    for y in range(ty, min(ty + trunk_height, WORLD_Y)):
        blocks[tx, y, tz] = BLOCK_WOOD
        
    leaf_bottom = ty + trunk_height - 2
    leaf_top = ty + trunk_height + 1
    for y in range(leaf_bottom, min(leaf_top + 1, WORLD_Y)):
        radius = 1 if y == leaf_top else 2
        for lx in range(tx - radius, tx + radius + 1):
            for lz in range(tz - radius, tz + radius + 1):
                if 0 <= lx < WORLD_X and 0 <= lz < WORLD_Z:
                    if abs(lx - tx) == radius and abs(lz - tz) == radius and random.random() < 0.4:
                        continue
                    if blocks[lx, y, lz] == BLOCK_AIR:
                        blocks[lx, y, lz] = BLOCK_LEAVES


def generate_world():
    # Multi-octave noise: n1 gives broad hill structures, n2 adds local variation
    n1 = value_noise_2d(WORLD_X, WORLD_Z, scale=16, seed=1)
    n2 = value_noise_2d(WORLD_X, WORLD_Z, scale=8, seed=2)
    
    # Combined height map with moderate hill variance
    height_map = n1 * 0.7 + n2 * 0.3

    blocks = np.zeros((WORLD_X, WORLD_Y, WORLD_Z), dtype=np.int8)
    for x in range(WORLD_X):
        for z in range(WORLD_Z):
            # Base height centered around 6, scaling up to ~20 for solid rolling hills
            h = int(6 + height_map[x, z] * 14)
            h = max(1, min(h, WORLD_Y - 1))
            
            blocks[x, 0, z] = BLOCK_BEDROCK
            blocks[x, 1:max(1, h - 3), z] = BLOCK_STONE

            # Coastal/water vs inland biome generation
            is_beach = h <= WATER_LEVEL + 1
            if is_beach:
                blocks[x, max(1, h - 3):h, z] = BLOCK_SAND
            else:
                blocks[x, max(1, h - 3):h - 1, z] = BLOCK_DIRT
                match random.randint(0, 5):
                    case 5:
                        blocks[x, h - 1, z] = BLOCK_AGOUTI
                    case _:
                        blocks[x, h - 1, z] = BLOCK_GRASS

            # Fill body of water up to WATER_LEVEL
            if h <= WATER_LEVEL:
                for y in range(h, WATER_LEVEL + 1):
                    blocks[x, y, z] = BLOCK_WATER

    tree_rng = random.Random(42)
    for x in range(2, WORLD_X - 2):
        for z in range(2, WORLD_Z - 2):
            ground_y = column_top(blocks, x, z)
            if ground_y > 0 and blocks[x, ground_y - 1, z] == BLOCK_GRASS:
                if tree_rng.random() < 0.02:
                    add_tree(blocks, x, ground_y, z)

    return blocks


def is_solid(blocks, x, y, z):
    nx, ny, nz = blocks.shape
    if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
        b = blocks[x, y, z]
        return b != BLOCK_AIR and b != BLOCK_WATER
    return False


def column_top(blocks, x, z):
    column = blocks[x, :, z]
    solid_ys = np.nonzero(column)[0]
    return int(solid_ys.max()) + 1 if len(solid_ys) else 0


# ============================================================================
#  Chunk Mesh Management
# ============================================================================
def build_chunk_mesh(blocks, atlas, cx, cz):
    x_start, x_end = cx * CHUNK_SIZE, min((cx + 1) * CHUNK_SIZE, WORLD_X)
    z_start, z_end = cz * CHUNK_SIZE, min((cz + 1) * CHUNK_SIZE, WORLD_Z)
    opaque_verts = []
    water_verts = []

    for x in range(x_start, x_end):
        for y in range(WORLD_Y):
            for z in range(z_start, z_end):
                block_id = blocks[x, y, z]
                if block_id == BLOCK_AIR:
                    continue
                
                layer = atlas.get_layer(block_id)
                for (dx, dy, dz), brightness, corners in FACES:
                    neighbor_x, neighbor_y, neighbor_z = x + dx, y + dy, z + dz
                    
                    if block_id == BLOCK_WATER:
                        # Water surfaces only show next to AIR
                        if 0 <= neighbor_x < WORLD_X and 0 <= neighbor_y < WORLD_Y and 0 <= neighbor_z < WORLD_Z:
                            if blocks[neighbor_x, neighbor_y, neighbor_z] != BLOCK_AIR:
                                continue
                    elif is_solid(blocks, neighbor_x, neighbor_y, neighbor_z):
                        continue

                    quad = [(x + cx_off, y + cy_off, z + cz_off) for cx_off, cy_off, cz_off in corners]
                    for idx in (0, 1, 2, 0, 2, 3):
                        vx, vy, vz = quad[idx]
                        u, v = UV_QUAD[idx]
                        target_list = water_verts if block_id == BLOCK_WATER else opaque_verts
                        target_list.extend((vx, vy, vz, u, v, brightness, layer))

    # Append transparent water quads at the end of the mesh buffer
    all_verts = opaque_verts + water_verts
    return np.array(all_verts, dtype="f4"), len(opaque_verts) // 7, len(water_verts) // 7


class ChunkManager:
    def __init__(self, ctx, prog, blocks, atlas):
        self.ctx = ctx
        self.prog = prog
        self.blocks = blocks
        self.atlas = atlas
        self.chunks = {}
        self.num_chunks_x = (WORLD_X + CHUNK_SIZE - 1) // CHUNK_SIZE
        self.num_chunks_z = (WORLD_Z + CHUNK_SIZE - 1) // CHUNK_SIZE
        self.rebuild_all()

    def rebuild_all(self):
        for cx in range(self.num_chunks_x):
            for cz in range(self.num_chunks_z):
                self.rebuild_chunk(cx, cz)

    def rebuild_chunk(self, cx, cz):
        if (cx, cz) in self.chunks:
            vbo, vao, o_cnt, w_cnt = self.chunks[(cx, cz)]
            if vbo: vbo.release()
            if vao: vao.release()

        vertex_data, opaque_count, water_count = build_chunk_mesh(self.blocks, self.atlas, cx, cz)
        if len(vertex_data) > 0:
            vbo = self.ctx.buffer(vertex_data.tobytes())
            vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 2f 1f 1f", "in_position", "in_uv", "in_brightness", "in_tile_idx")])
            self.chunks[(cx, cz)] = (vbo, vao, opaque_count, water_count)
        else:
            self.chunks[(cx, cz)] = (None, None, 0, 0)

    def update_block(self, x, y, z):
        cx, cz = x // CHUNK_SIZE, z // CHUNK_SIZE
        self.rebuild_chunk(cx, cz)

        if x % CHUNK_SIZE == 0 and cx > 0:
            self.rebuild_chunk(cx - 1, cz)
        if x % CHUNK_SIZE == CHUNK_SIZE - 1 and cx < self.num_chunks_x - 1:
            self.rebuild_chunk(cx + 1, cz)
        if z % CHUNK_SIZE == 0 and cz > 0:
            self.rebuild_chunk(cx, cz - 1)
        if z % CHUNK_SIZE == CHUNK_SIZE - 1 and cz < self.num_chunks_z - 1:
            self.rebuild_chunk(cx, cz + 1)

    def render_opaque(self):
        for vbo, vao, opaque_count, _ in self.chunks.values():
            if vao is not None and opaque_count > 0:
                vao.render(moderngl.TRIANGLES, vertices=opaque_count)

    def render_transparent(self):
        for vbo, vao, opaque_count, water_count in self.chunks.values():
            if vao is not None and water_count > 0:
                vao.render(moderngl.TRIANGLES, vertices=water_count, first=opaque_count)


# ============================================================================
#  Raycasting & Physics
# ============================================================================
RAY_MAX_DISTANCE = 6.0
RAY_STEP = 0.05

def raycast(blocks, origin, direction):
    prev_cell = None
    dist = 0.0
    while dist <= RAY_MAX_DISTANCE:
        point = origin + direction * dist
        cell = (math.floor(point[0]), math.floor(point[1]), math.floor(point[2]))
        if is_solid(blocks, *cell):
            return cell, prev_cell
        prev_cell = cell
        dist += RAY_STEP
    return None, None


# ============================================================================
#  Inventory & Crafting System
# ============================================================================
NUM_SLOTS = 10

@dataclass
class Slot:
    block_id: int = BLOCK_AIR
    count: int = 0

@dataclass
class Inventory:
    slots: list = field(default_factory=lambda: [Slot() for _ in range(NUM_SLOTS)])
    crafting_grid: list = field(default_factory=lambda: [Slot() for _ in range(4)])
    crafting_result: Slot = field(default_factory=Slot)
    cursor_slot: Slot = field(default_factory=Slot)
    selected_index: int = 0

    @property
    def selected_slot(self):
        return self.slots[self.selected_index]

    def add(self, block_id, amount=1):
        for slot in self.slots:
            if slot.block_id == block_id:
                slot.count += amount
                return True
        for slot in self.slots:
            if slot.block_id == BLOCK_AIR:
                slot.block_id = block_id
                slot.count = amount
                return True
        return False

    def try_take_selected(self, amount=1):
        slot = self.selected_slot
        if slot.block_id != BLOCK_AIR and slot.count >= amount:
            slot.count -= amount
            if slot.count == 0:
                slot.block_id = BLOCK_AIR
            return True
        return False

    def check_crafting_recipes(self):
        g = [s.block_id for s in self.crafting_grid]

        # Recipe: 1 Wood -> 4 Planks
        wood_count = sum(1 for b in g if b == BLOCK_WOOD)
        other_count = sum(1 for b in g if b != BLOCK_WOOD and b != BLOCK_AIR)
        if wood_count == 1 and other_count == 0:
            self.crafting_result.block_id = BLOCK_PLANKS
            self.crafting_result.count = 4
            return

        self.crafting_result.block_id = BLOCK_AIR
        self.crafting_result.count = 0

    def consume_crafting_ingredients(self):
        for s in self.crafting_grid:
            if s.block_id != BLOCK_AIR:
                s.count -= 1
                if s.count <= 0:
                    s.block_id = BLOCK_AIR
                    s.count = 0
        self.check_crafting_recipes()


def cell_overlaps_aabb(cell, min_corner, max_corner):
    x, y, z = cell
    return (
        x < max_corner[0] and x + 1 > min_corner[0]
        and y < max_corner[1] and y + 1 > min_corner[1]
        and z < max_corner[2] and z + 1 > min_corner[2]
    )


def break_and_mine_block(blocks, inventory, cell):
    x, y, z = cell
    block_id = blocks[x, y, z]
    if block_id == BLOCK_AIR:
        return False
    blocks[x, y, z] = BLOCK_AIR
    inventory.add(block_id)
    return True


def try_place_block(blocks, inventory, cell, player_pos):
    if cell is None:
        return False
    x, y, z = cell
    nx, ny, nz = blocks.shape
    if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
        return False
    if blocks[x, y, z] != BLOCK_AIR and blocks[x, y, z] != BLOCK_WATER:
        return False

    min_corner, max_corner = player_aabb(player_pos)
    if cell_overlaps_aabb(cell, min_corner, max_corner):
        return False

    selected_slot = inventory.selected_slot
    if selected_slot.block_id == BLOCK_AIR:
        return False

    block_id = selected_slot.block_id
    if not inventory.try_take_selected(1):
        return False

    blocks[x, y, z] = block_id
    return True


# ============================================================================
#  Block destruction & Selection visuals
# ============================================================================
def build_overlay_block_mesh(blocks, atlas, cell):
    x, y, z = cell
    block_id = blocks[x, y, z]
    if block_id == BLOCK_AIR:
        return np.array([], dtype="f4")

    layer = atlas.get_layer(block_id)
    verts = []
    inflate = 0.001

    for (dx, dy, dz), brightness, corners in FACES:
        quad = [
            (x + cx * (1 + 2 * inflate) - inflate,
             y + cy * (1 + 2 * inflate) - inflate,
             z + cz * (1 + 2 * inflate) - inflate)
            for cx, cy, cz in corners
        ]
        for idx in (0, 1, 2, 0, 2, 3):
            vx, vy, vz = quad[idx]
            u, v = UV_QUAD[idx]
            verts.extend((vx, vy, vz, u, v, layer))

    return np.array(verts, dtype="f4")


def build_block_outline_mesh(cell):
    x, y, z = cell
    eps = 0.002  # Inflate slightly to eliminate Z-fighting against faces
    x0, x1 = x - eps, x + 1 + eps
    y0, y1 = y - eps, y + 1 + eps
    z0, z1 = z - eps, z + 1 + eps

    # 12 edges (2 vertices per line = 24 points)
    edges = [
        # Bottom square
        (x0, y0, z0), (x1, y0, z0),
        (x1, y0, z0), (x1, y0, z1),
        (x1, y0, z1), (x0, y0, z1),
        (x0, y0, z1), (x0, y0, z0),
        # Top square
        (x0, y1, z0), (x1, y1, z0),
        (x1, y1, z0), (x1, y1, z1),
        (x1, y1, z1), (x0, y1, z1),
        (x0, y1, z1), (x0, y1, z0),
        # Vertical edges connecting top and bottom
        (x0, y0, z0), (x0, y1, z0),
        (x1, y0, z0), (x1, y1, z0),
        (x1, y0, z1), (x1, y1, z1),
        (x0, y0, z1), (x0, y1, z1),
    ]

    verts = []
    for p in edges:
        verts.extend(p)
    return np.array(verts, dtype="f4")


# ============================================================================
#  Player Physics & Camera
# ============================================================================
_HALF_WIDTH = PLAYER_WIDTH / 2
_AXIS_OFFSETS = (
    (-_HALF_WIDTH, _HALF_WIDTH),
    (0.0, PLAYER_HEIGHT),
    (-_HALF_WIDTH, _HALF_WIDTH),
)


def player_aabb(pos):
    min_corner = [pos[i] + _AXIS_OFFSETS[i][0] for i in range(3)]
    max_corner = [pos[i] + _AXIS_OFFSETS[i][1] for i in range(3)]
    return min_corner, max_corner


def aabb_overlaps_solid_block(blocks, min_corner, max_corner):
    eps = COLLISION_EPS
    bx0, bx1 = math.floor(min_corner[0]), math.floor(max_corner[0] - eps)
    by0, by1 = math.floor(min_corner[1]), math.floor(max_corner[1] - eps)
    bz0, bz1 = math.floor(min_corner[2]), math.floor(max_corner[2] - eps)
    for bx in range(bx0, bx1 + 1):
        for by in range(by0, by1 + 1):
            for bz in range(bz0, bz1 + 1):
                if is_solid(blocks, bx, by, bz):
                    return True
    return False


def move_player(blocks, pos, vel, dt):
    vel[1] += GRAVITY * dt
    on_ground = False

    for axis in range(3):
        delta = vel[axis] * dt
        if delta == 0.0:
            continue

        pos[axis] += delta
        min_corner, max_corner = player_aabb(pos)

        if aabb_overlaps_solid_block(blocks, min_corner, max_corner):
            min_off, max_off = _AXIS_OFFSETS[axis]
            if delta > 0:
                boundary = math.floor(pos[axis] + max_off)
                pos[axis] = boundary - max_off - COLLISION_EPS
            else:
                boundary = math.floor(pos[axis] + min_off) + 1
                pos[axis] = boundary - min_off + COLLISION_EPS
                if axis == 1:
                    on_ground = True
            vel[axis] = 0.0

    return on_ground


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def perspective(fov_y_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2)
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.identity(4, dtype="f4")
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def upload_matrix(uniform, m):
    uniform.write(m.T.astype("f4").tobytes())


def facing_direction(yaw, pitch):
    x = math.cos(pitch) * math.sin(yaw)
    y = math.sin(pitch)
    z = -math.cos(pitch) * math.cos(yaw)
    return normalize(np.array([x, y, z], dtype="f4"))


@dataclass
class Player:
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype="f4"))
    yaw: float = 0.0
    pitch: float = 0.0
    on_ground: bool = False

    def eye_pos(self):
        return self.pos + np.array([0, EYE_HEIGHT, 0], dtype="f4")


def handle_mouse_look(player, rel_x, rel_y):
    player.yaw += rel_x * MOUSE_SENS
    player.pitch -= rel_y * MOUSE_SENS
    pitch_limit = math.pi / 2 - 0.01
    player.pitch = max(-pitch_limit, min(pitch_limit, player.pitch))


def handle_walk_input(player, forward, keys):
    flat_forward = normalize(np.array([forward[0], 0, forward[2]], dtype="f4"))
    right = normalize(np.cross(flat_forward, np.array([0, 1, 0], dtype="f4")))

    move = np.zeros(3, dtype="f4")
    if keys[pygame.K_w]:
        move += flat_forward
    if keys[pygame.K_s]:
        move -= flat_forward
    if keys[pygame.K_d]:
        move += right
    if keys[pygame.K_a]:
        move -= right
    if np.linalg.norm(move) > 0:
        move = normalize(move)

    player.vel[0] = move[0] * WALK_SPEED
    player.vel[2] = move[2] * WALK_SPEED

    if keys[pygame.K_SPACE] and player.on_ground:
        player.vel[1] = JUMP_SPEED


# ============================================================================
#  2D HUD & Crafting Overlay Setup
# ============================================================================
HUD_VERT_SHADER = """
#version 330
uniform vec2 screen_size;
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    vec2 ndc = vec2(
        (in_pos.x / screen_size.x) * 2.0 - 1.0,
        1.0 - (in_pos.y / screen_size.y) * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
}
"""

HUD_FRAG_SHADER = """
#version 330
uniform sampler2D tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(tex, v_uv);
}
"""

SLOT_SIZE = 54
HUD_MARGIN = 20

ui_rects = {
    'hotbar': [],
    'crafting': [],
    'result': None
}

loaded_ui_textures = {}

def get_ui_texture(block_id):
    if block_id not in loaded_ui_textures:
        filename = TEXTURE_FILES.get(block_id, None)
        if filename:
            try:
                img_path = resource_path(filename)
                img = pygame.image.load(img_path).convert_alpha()
                img = pygame.transform.scale(img, (SLOT_SIZE - 10, SLOT_SIZE - 10))
                loaded_ui_textures[block_id] = img
            except Exception as e:
                print(f"Warning: Could not load UI texture {filename}: {e}")
                loaded_ui_textures[block_id] = None
        else:
            loaded_ui_textures[block_id] = None
    return loaded_ui_textures[block_id]


def draw_slot(surf, rect, slot, font, border_color=(15, 15, 15)):
    fill = (50, 50, 50, 180)
    pygame.draw.rect(surf, fill, rect)
    pygame.draw.rect(surf, border_color, rect, width=3)

    if slot.block_id != BLOCK_AIR:
        icon = get_ui_texture(slot.block_id)
        if icon:
            surf.blit(icon, (rect.x + 5, rect.y + 5))

    if slot.count > 0:
        count_label = font.render(str(slot.count), True, (255, 255, 255))
        surf.blit(count_label, (rect.x + 4, rect.y + SLOT_SIZE - 20))


def render_ui_surface(inventory, crafting_open, mouse_pos):
    font = pygame.font.SysFont(None, 20)
    surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    ui_rects['hotbar'].clear()
    ui_rects['crafting'].clear()

    # 1. Hotbar Rendering
    hotbar_w = SLOT_SIZE * NUM_SLOTS
    hotbar_x = (WIDTH - hotbar_w) // 2
    hotbar_y = HEIGHT - HUD_MARGIN - SLOT_SIZE

    for i, slot in enumerate(inventory.slots):
        rect = pygame.Rect(hotbar_x + i * SLOT_SIZE, hotbar_y, SLOT_SIZE, SLOT_SIZE)
        ui_rects['hotbar'].append(rect)

        is_selected = i == inventory.selected_index
        border = (255, 255, 0) if is_selected else (15, 15, 15)
        draw_slot(surf, rect, slot, font, border)

        slot_text = "0" if i == 9 else str(i + 1)
        key_label = font.render(slot_text, True, (255, 255, 255))
        surf.blit(key_label, (rect.x + 4, rect.y + 4))

    # 2. Crafting Overlay Rendering
    if crafting_open:
        panel_w, panel_h = 320, 220
        panel_x, panel_y = (WIDTH - panel_w) // 2, (HEIGHT - panel_h) // 2 - 40
        pygame.draw.rect(surf, (30, 30, 30, 230), (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(surf, (200, 200, 200), (panel_x, panel_y, panel_w, panel_h), width=3)

        title = font.render("Agoufting Table", True, (255, 255, 255))
        surf.blit(title, (panel_x + 15, panel_y + 15))

        grid_start_x, grid_start_y = panel_x + 30, panel_y + 50
        for row in range(2):
            for col in range(2):
                idx = row * 2 + col
                rect = pygame.Rect(grid_start_x + col * (SLOT_SIZE + 5),
                                   grid_start_y + row * (SLOT_SIZE + 5),
                                   SLOT_SIZE, SLOT_SIZE)
                ui_rects['crafting'].append(rect)
                draw_slot(surf, rect, inventory.crafting_grid[idx], font)

        arrow_label = font.render("-->", True, (255, 255, 255))
        surf.blit(arrow_label, (panel_x + 165, panel_y + 80))

        result_rect = pygame.Rect(panel_x + 210, panel_y + 65, SLOT_SIZE, SLOT_SIZE)
        ui_rects['result'] = result_rect
        draw_slot(surf, result_rect, inventory.crafting_result, font, border_color=(0, 255, 0))
    else:
        # 3. Simple Dot Crosshair
        cx, cy = WIDTH // 2, HEIGHT // 2
        pygame.draw.circle(surf, (0, 0, 0, 200), (cx, cy), 3)        # Outer border
        pygame.draw.circle(surf, (255, 255, 255, 230), (cx, cy), 2)  # Inner dot

    # 4. Item Held by Cursor
    if inventory.cursor_slot.block_id != BLOCK_AIR:
        mx, my = mouse_pos
        cursor_rect = pygame.Rect(mx - SLOT_SIZE // 2, my - SLOT_SIZE // 2, SLOT_SIZE, SLOT_SIZE)
        draw_slot(surf, cursor_rect, inventory.cursor_slot, font, border_color=(255, 255, 255))

    return surf


def surface_to_texture(ctx, surface):
    data = pygame.image.tostring(surface, "RGBA", False)
    texture = ctx.texture(surface.get_size(), 4, data)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return texture


def build_hud_quad():
    return np.array([
        0,     0,      0, 0,
        WIDTH, 0,      1, 0,
        WIDTH, HEIGHT, 1, 1,
        0,     0,      0, 0,
        WIDTH, HEIGHT, 1, 1,
        0,     HEIGHT, 0, 1,
    ], dtype="f4")


# ============================================================================
#  Main Window Initialization & Game Loop
# ============================================================================
def create_window():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(f"agouticraft {ACversion}")
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)


def make_spawn_player(blocks):
    spawn_x, spawn_z = WORLD_X // 2, WORLD_Z // 2
    ground_y = column_top(blocks, spawn_x, spawn_z)
    spawn_pos = np.array([spawn_x, max(ground_y, WATER_LEVEL + 1) + 3.0, spawn_z], dtype="f4")
    return Player(pos=spawn_pos, yaw=math.radians(45), pitch=math.radians(-15))


HOTBAR_KEYS = {
    pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3, pygame.K_5: 4,
    pygame.K_6: 5, pygame.K_7: 6, pygame.K_8: 7, pygame.K_9: 8, pygame.K_0: 9,
}


def handle_slot_left_click(target_slot, cursor_slot):
    if cursor_slot.block_id == BLOCK_AIR:
        cursor_slot.block_id = target_slot.block_id
        cursor_slot.count = target_slot.count
        target_slot.block_id = BLOCK_AIR
        target_slot.count = 0
    elif target_slot.block_id == BLOCK_AIR:
        target_slot.block_id = cursor_slot.block_id
        target_slot.count = cursor_slot.count
        cursor_slot.block_id = BLOCK_AIR
        cursor_slot.count = 0
    elif target_slot.block_id == cursor_slot.block_id:
        target_slot.count += cursor_slot.count
        cursor_slot.block_id = BLOCK_AIR
        cursor_slot.count = 0
    else:
        target_slot.block_id, cursor_slot.block_id = cursor_slot.block_id, target_slot.block_id
        target_slot.count, cursor_slot.count = cursor_slot.count, target_slot.count


def handle_slot_right_click(target_slot, cursor_slot):
    if cursor_slot.block_id != BLOCK_AIR:
        if target_slot.block_id == BLOCK_AIR:
            target_slot.block_id = cursor_slot.block_id
            target_slot.count = 1
            cursor_slot.count -= 1
        elif target_slot.block_id == cursor_slot.block_id:
            target_slot.count += 1
            cursor_slot.count -= 1

        if cursor_slot.count <= 0:
            cursor_slot.block_id = BLOCK_AIR
            cursor_slot.count = 0

    elif target_slot.block_id != BLOCK_AIR:
        take_count = math.ceil(target_slot.count / 2.0)
        cursor_slot.block_id = target_slot.block_id
        cursor_slot.count = take_count
        target_slot.count -= take_count

        if target_slot.count <= 0:
            target_slot.block_id = BLOCK_AIR
            target_slot.count = 0


def main():
    def bf(txt):
        print(txt, end="", flush=True)
    def dn():
        print(" Done")

    bf("Creating window...")
    create_window()
    dn()

    bf("Creating moderngl context...")
    ctx = moderngl.create_context()
    dn()
    bf("Enabling moderngl depth test...")
    ctx.enable(moderngl.DEPTH_TEST)
    dn()

    bf("Loading textures...")
    atlas = TextureAtlas(ctx)
    dn()
    bf("Generating world...")
    blocks = generate_world()
    dn()

    prog = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER)
    mvp_uniform = prog["mvp"]
    prog["atlas"].value = 0
    bf("Creating chunk manager...")
    chunk_manager = ChunkManager(ctx, prog, blocks, atlas)
    dn()

    overlay_prog = ctx.program(vertex_shader=OVERLAY_VERT_SHADER, fragment_shader=OVERLAY_FRAG_SHADER)
    overlay_mvp = overlay_prog["mvp"]
    overlay_dark_factor = overlay_prog["dark_factor"]
    overlay_prog["atlas"].value = 0
    
    break_vbo = ctx.buffer(reserve=36 * 6 * 4, dynamic=True) # 6 floats per vertex (3f 2f 1f)
    break_vao = ctx.vertex_array(overlay_prog, [(break_vbo, "3f 2f 1f", "in_position", "in_uv", "in_tile_idx")])

    line_prog = ctx.program(vertex_shader=LINE_VERT_SHADER, fragment_shader=LINE_FRAG_SHADER)
    line_mvp = line_prog["mvp"]
    line_vbo = ctx.buffer(reserve=24 * 3 * 4, dynamic=True) # 24 verts * 3f
    line_vao = ctx.vertex_array(line_prog, [(line_vbo, "3f", "in_position")])

    hud_prog = ctx.program(vertex_shader=HUD_VERT_SHADER, fragment_shader=HUD_FRAG_SHADER)
    hud_prog["screen_size"].value = (WIDTH, HEIGHT)
    
    bf("Creating inventory...")
    inventory = Inventory()
    crafting_open = False
    dn()

    hud_texture = surface_to_texture(ctx, render_ui_surface(inventory, crafting_open, (0, 0)))
    hud_vbo = ctx.buffer(build_hud_quad().tobytes())
    hud_vao = ctx.vertex_array(hud_prog, [(hud_vbo, "2f 2f", "in_pos", "in_uv")])
    hud_dirty = False

    bf("Creating player...")
    player = make_spawn_player(blocks)
    clock = pygame.time.Clock()
    proj = perspective(FOV_Y, WIDTH / HEIGHT, NEAR, FAR)
    dn()

    mining_cell = None
    mining_progress = 0.0

    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        mouse_pos = pygame.mouse.get_pos()

        place_clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in (pygame.K_e, pygame.K_c):
                    crafting_open = not crafting_open
                    pygame.event.set_grab(not crafting_open)
                    pygame.mouse.set_visible(crafting_open)
                    hud_dirty = True
                elif event.key in HOTBAR_KEYS:
                    inventory.selected_index = HOTBAR_KEYS[event.key]
                    hud_dirty = True

            elif event.type == pygame.MOUSEMOTION:
                if not crafting_open:
                    handle_mouse_look(player, *event.rel)
                else:
                    hud_dirty = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if crafting_open:
                    if event.button == 1:
                        for i, rect in enumerate(ui_rects['hotbar']):
                            if rect.collidepoint(mouse_pos):
                                handle_slot_left_click(inventory.slots[i], inventory.cursor_slot)
                                hud_dirty = True

                        for i, rect in enumerate(ui_rects['crafting']):
                            if rect.collidepoint(mouse_pos):
                                handle_slot_left_click(inventory.crafting_grid[i], inventory.cursor_slot)
                                inventory.check_crafting_recipes()
                                hud_dirty = True

                        if ui_rects['result'] and ui_rects['result'].collidepoint(mouse_pos):
                            res = inventory.crafting_result
                            if res.block_id != BLOCK_AIR:
                                if inventory.cursor_slot.block_id == BLOCK_AIR:
                                    inventory.cursor_slot.block_id = res.block_id
                                    inventory.cursor_slot.count = res.count
                                    inventory.consume_crafting_ingredients()
                                    hud_dirty = True
                                elif inventory.cursor_slot.block_id == res.block_id:
                                    inventory.cursor_slot.count += res.count
                                    inventory.consume_crafting_ingredients()
                                    hud_dirty = True

                    elif event.button == 3:
                        for i, rect in enumerate(ui_rects['hotbar']):
                            if rect.collidepoint(mouse_pos):
                                handle_slot_right_click(inventory.slots[i], inventory.cursor_slot)
                                hud_dirty = True

                        for i, rect in enumerate(ui_rects['crafting']):
                            if rect.collidepoint(mouse_pos):
                                handle_slot_right_click(inventory.crafting_grid[i], inventory.cursor_slot)
                                inventory.check_crafting_recipes()
                                hud_dirty = True

                        if ui_rects['result'] and ui_rects['result'].collidepoint(mouse_pos):
                            res = inventory.crafting_result
                            if res.block_id != BLOCK_AIR:
                                if inventory.cursor_slot.block_id == BLOCK_AIR:
                                    inventory.cursor_slot.block_id = res.block_id
                                    inventory.cursor_slot.count = res.count
                                    inventory.consume_crafting_ingredients()
                                    hud_dirty = True
                                elif inventory.cursor_slot.block_id == res.block_id:
                                    inventory.cursor_slot.count += res.count
                                    inventory.consume_crafting_ingredients()
                                    hud_dirty = True
                else:
                    if event.button == 3:
                        place_clicked = True

        if not crafting_open:
            forward = facing_direction(player.yaw, player.pitch)
            handle_walk_input(player, forward, pygame.key.get_pressed())
            player.on_ground = move_player(blocks, player.pos, player.vel, dt)
            hit_cell, place_cell = raycast(blocks, player.eye_pos(), forward)
        else:
            hit_cell, place_cell = None, None

        # Block Mining
        mouse_buttons = pygame.mouse.get_pressed()
        is_left_clicking = mouse_buttons[0]

        if not crafting_open and is_left_clicking and hit_cell is not None:
            if hit_cell != mining_cell:
                mining_cell = hit_cell
                mining_progress = 0.0

            x, y, z = mining_cell
            block_type = blocks[x, y, z]
            break_time = BLOCK_BREAK_TIMES.get(block_type, 1.0)

            if break_time != float('inf'):
                mining_progress += dt / break_time
                if mining_progress >= 1.0:
                    if break_and_mine_block(blocks, inventory, mining_cell):
                        chunk_manager.update_block(*mining_cell)
                        hud_dirty = True
                    mining_cell = None
                    mining_progress = 0.0
        else:
            mining_cell = None
            mining_progress = 0.0

        if not crafting_open and place_clicked and try_place_block(blocks, inventory, place_cell, player.pos):
            chunk_manager.update_block(*place_cell)
            hud_dirty = True

        if hud_dirty:
            hud_texture.release()
            hud_texture = surface_to_texture(ctx, render_ui_surface(inventory, crafting_open, mouse_pos))
            hud_dirty = False

        forward = facing_direction(player.yaw, player.pitch)
        view = look_at(player.eye_pos(), player.eye_pos() + forward, np.array([0, 1, 0], dtype="f4"))
        mvp_matrix = proj @ view

        upload_matrix(mvp_uniform, mvp_matrix)

        ctx.clear(0.55, 0.75, 0.95)
        
        # 1. Render Solid Terrain Opaque
        atlas.texture_array.use(location=0)
        chunk_manager.render_opaque()

        # 2. Render Transparent Water Layers with Alpha Blending enabled
        ctx.enable(moderngl.BLEND)
        chunk_manager.render_transparent()
        ctx.disable(moderngl.BLEND)

        # Render Mining Animation Overlay
        if mining_cell is not None and mining_progress > 0.0:
            overlay_mesh = build_overlay_block_mesh(blocks, atlas, mining_cell)
            if len(overlay_mesh) > 0:
                upload_matrix(overlay_mvp, mvp_matrix)
                overlay_dark_factor.value = float(mining_progress)
                break_vbo.write(overlay_mesh.tobytes())
                break_vao.render(moderngl.TRIANGLES)

        # Render Block Selection Wireframe Outline
        if hit_cell is not None and not crafting_open:
            outline_mesh = build_block_outline_mesh(hit_cell)
            if len(outline_mesh) > 0:
                upload_matrix(line_mvp, mvp_matrix)
                line_vbo.write(outline_mesh.tobytes())
                line_vao.render(moderngl.LINES)

        # Render 2D UI Overlay
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        hud_texture.use(location=1)
        hud_prog["tex"].value = 1
        hud_vao.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "="*50, file=sys.stderr)
        print("CRITICAL RUNTIME ERROR CATCH:", file=sys.stderr)
        print("="*50, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("="*50 + "\n", file=sys.stderr)
        input("Press ENTER to exit...")