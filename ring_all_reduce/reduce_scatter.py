# %%manim -v WARNING -qm ReduceScatterAnimation

from manim import *
import numpy as np

class ReduceScatterAnimation(Scene):
    def construct(self):
        # Match the pastel colors from the reference image, wrapped in ManimColor to fix the interpolate error
        chunk_colors = [
            ManimColor("#5D9CEC"), # Blue (C0)
            ManimColor("#A8E6CF"), # Light Green (C1)
            ManimColor("#D98880"), # Pink/Red (C2)
            ManimColor("#F9E79F"), # Yellow (C3)
            ManimColor("#76D7C4"), # Teal (C4)
            ManimColor("#C39BD3")  # Purple (C5)
        ]
        
        # Base color for the starting state (opaque white)
        base_color = ManimColor("#FFFFFF")
        
        # 1. Title
        title = Text("Reduce-scatter", font_size=32, weight=BOLD, font="Arial")
        title.to_corner(UL, buff=0.5)
        self.add(title)

        # 2. Layout Configuration
        num_gpus = 6
        radius = 2.8
        center_offset = DOWN * 0.2

        # 3. Dynamic Center Text
        step_text = Text("Reduce-Scatter: Initial State", font_size=16, color=WHITE, weight=BOLD, font="Arial")
        step_bg = SurroundingRectangle(step_text, color=WHITE, stroke_width=1, fill_color="#1E3232", fill_opacity=1, corner_radius=0.1)
        center_group = VGroup(step_bg, step_text).move_to(center_offset)
        
        sub_text = Text("All 6 chunks calculated per GPU (1/6)", font_size=12, color=LIGHT_GRAY, font="Arial")
        sub_text.next_to(center_group, DOWN, buff=0.2)
        
        self.add(center_group, sub_text)

        # 4. Create GPUs and Initial Shard States
        gpus = VGroup()
        gpu_blocks = {}
        
        for i in range(num_gpus):
            # Arranged clockwise
            angle = np.pi/2 - i * (2 * np.pi / num_gpus) 
            
            rects = VGroup()
            labels = VGroup()
            fracs = VGroup()
            blocks_for_this_gpu = []
            
            for j in range(6):
                rect = Rectangle(height=0.2, width=0.9, stroke_width=1, stroke_color=WHITE)
                
                # Interpolate from opaque white to target color. 
                # Alpha is 1/6 so it starts mostly white (very light). Opacity is kept 100% solid.
                start_color = interpolate_color(base_color, chunk_colors[j], 1/6.0)
                rect.set_fill(start_color, opacity=1.0)
                
                rects.add(rect)
                
            rects.arrange(DOWN, buff=0)
            
            for j in range(6):
                rect = rects[j]
                
                # Outer label (C0, C1, etc.) stays white as it's on the black background
                c_label = Text(f"C{j}", font_size=10, color=WHITE, font="Arial")
                c_label.next_to(rect, LEFT, buff=0.1)
                labels.add(c_label)
                
                # Inner fraction label changed to BLACK for contrast against the white/pastel backgrounds
                frac = Text("1/6", font_size=10, color=BLACK, font="Arial")
                frac.move_to(rect.get_right() + LEFT * 0.2)
                fracs.add(frac)
                
                blocks_for_this_gpu.append({'rect': rect, 'frac': frac})
            
            box = SurroundingRectangle(VGroup(rects, labels), buff=0.1, color=WHITE, stroke_width=1.2, corner_radius=0.05)
            
            gpu_label = Text(f"GPU {i}", font_size=14, weight=BOLD, font="Arial")
            gpu_label.next_to(box, DOWN, buff=0.1)
            
            gpu_node = VGroup(box, rects, labels, fracs, gpu_label)
            gpu_node.move_to(center_offset + np.array([radius * np.cos(angle), radius * np.sin(angle), 0]))
            
            gpus.add(gpu_node)
            gpu_blocks[i] = blocks_for_this_gpu
            
        self.add(gpus)

        # 5. Create Directional Arrows
        arrows = VGroup()
        for i in range(num_gpus):
            start_gpu = gpus[i][0] # The bounding box
            end_gpu = gpus[(i + 1) % num_gpus][0] 
            
            direction = end_gpu.get_center() - start_gpu.get_center()
            
            # Explicitly force vertical connections for GPUs 1->2 and 4->5
            if i == 1:
                start_pt = start_gpu.get_bottom()
                end_pt = end_gpu.get_top()
            elif i == 4:
                start_pt = start_gpu.get_top()
                end_pt = end_gpu.get_bottom()
            else:
                start_pt = start_gpu.get_boundary_point(direction)
                end_pt = end_gpu.get_boundary_point(-direction)
            
            arrow = Arrow(
                start_pt, 
                end_pt, 
                buff=0.15, 
                color=LIGHT_GRAY,
                stroke_width=3,
                tip_length=0.15
            )
            arrows.add(arrow)
        self.add(arrows)

        # Pause to view the initial state
        self.wait(1.5)

        # 6. Animation Loop: The 5 steps of Reduce-Scatter
        for s in range(1, 6):
            new_step_text = Text(f"Reduce-Scatter : Step {s} / 5", font_size=16, color=WHITE, weight=BOLD, font="Arial")
            new_step_text.move_to(step_text.get_center())
            self.play(Transform(step_text, new_step_text), run_time=0.5)
            
            moving_shards = VGroup()
            dest_updates = []
            
            for i in range(num_gpus):
                target_gpu = (i + 1) % num_gpus
                chunk_id = (i - s + 1) % 6 
                
                source_rect = gpu_blocks[i][chunk_id]['rect']
                
                # Copying retains the current chunk's color state
                moving_rect = source_rect.copy()
                moving_shards.add(moving_rect)
                
                # The updated count for the target chunk
                new_count = s + 1
                dest_updates.append((target_gpu, chunk_id, new_count))
                
            self.add(moving_shards)
            
            move_anims = []
            for i, shard in enumerate(moving_shards):
                target_gpu_idx, chunk_id, _ = dest_updates[i]
                target_rect = gpu_blocks[target_gpu_idx][chunk_id]['rect']
                move_anims.append(shard.animate.move_to(target_rect.get_center()))
                
            self.play(*move_anims, run_time=1.5)
            
            update_anims = []
            for target_gpu, chunk_id, new_count in dest_updates:
                rect = gpu_blocks[target_gpu][chunk_id]['rect']
                frac = gpu_blocks[target_gpu][chunk_id]['frac']
                
                # Calculate the new interpolated color progressing towards the target color
                new_color = interpolate_color(base_color, chunk_colors[chunk_id], new_count / 6.0)
                
                # If it's the final reduced chunk (6/6), add a distinct black border
                if new_count == 6:
                    update_anims.append(rect.animate.set_fill(new_color, opacity=1.0).set_stroke(color=BLACK, width=2.5))
                else:
                    update_anims.append(rect.animate.set_fill(new_color, opacity=1.0))
                
                # Keep the inner text black for consistent visibility on light backgrounds
                new_frac = Text(f"{new_count}/6", font_size=10, color=BLACK, font="Arial")
                new_frac.move_to(rect.get_right() + LEFT * 0.2)
                
                update_anims.append(Transform(frac, new_frac))
                
            self.play(*update_anims, FadeOut(moving_shards), run_time=0.5)
            self.wait(0.5)

        self.wait(3)