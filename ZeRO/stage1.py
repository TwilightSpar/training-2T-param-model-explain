from manim import *
import numpy as np

class ZeROStage1(Scene):
    def construct(self):
        def get_color(comp, status, c, gpu_idx):
            W_BASE = "#5D9CEC"
            W_UPD = "#2E86C1"
            G_BASE = "#D98880"
            G_UPD = "#E74C3C"
            O_BASE = "#A8E6CF"
            O_UPD = "#27AE60"
            EMPTY = "#555555"
            
            palette = [
                [W_BASE, W_UPD],
                [G_BASE, G_UPD],
                [O_BASE, O_UPD]
            ]
            base_col, upd_col = palette[comp]
            
            if status == 0: return EMPTY
            elif status == 1: return base_col if c == gpu_idx else EMPTY
            elif status == 2: return base_col
            elif status == 3: return upd_col if c == gpu_idx else EMPTY
            elif status == 4: return upd_col
            elif status == 5: return upd_col if c == gpu_idx else base_col
            return EMPTY

        def create_gpus(state_w, state_g, state_o):
            gpus = VGroup()
            for i in range(6):
                blocks_group = VGroup()
                letters_group = VGroup()
                
                for comp_idx, (name, status_list, rect_h) in enumerate([
                    ("W", state_w[i], 0.25), 
                    ("G", state_g[i], 0.25), 
                    ("O", state_o[i], 0.45)
                ]):
                    letter = MarkupText(name, font_size=12, weight=BOLD, font="Arial")
                    layers = VGroup()
                    
                    for L in range(4):
                        layer_status = status_list[L]
                        chunks = VGroup()
                        for c in range(6):
                            sq = Rectangle(height=rect_h, width=0.12, stroke_width=0.5, stroke_color=WHITE)
                            color = get_color(comp_idx, layer_status, c, i)
                            sq.set_fill(color, 1.0)
                            chunks.add(sq)
                        chunks.arrange(RIGHT, buff=0)
                        
                        layer_border = SurroundingRectangle(chunks, buff=0, stroke_width=1.2, stroke_color=WHITE)
                        lbl = MarkupText(f"L{L+1}", font_size=10, color=WHITE).move_to(chunks.get_center())
                        
                        single_layer = VGroup(chunks, layer_border, lbl)
                        layers.add(single_layer)
                        
                    layers.arrange(RIGHT, buff=0.12)
                    blocks_group.add(layers)
                    letters_group.add(letter)
                    
                blocks_group.arrange(DOWN, buff=0.1)
                for letter, blocks in zip(letters_group, blocks_group):
                    letter.next_to(blocks, LEFT, buff=0.1)
                    
                gpu_content = VGroup(letters_group, blocks_group)
                box = SurroundingRectangle(gpu_content, buff=0.1, color=WHITE, stroke_width=1, corner_radius=0.05)
                title = MarkupText(f"GPU {i}", font_size=14, weight=BOLD, font="Arial").next_to(box, DOWN, buff=0.1)
                
                node = VGroup(box, gpu_content, title)
                angle = np.pi/2 - i * (2 * np.pi / 6)
                pos = DOWN*0.2 + np.array([3.1 * np.cos(angle), 3.1 * np.sin(angle), 0])
                
                if i in [2, 3, 4]:
                    pos += UP * 0.4
                    
                node.move_to(pos)
                gpus.add(node)
                
            return gpus

        def animate_transition(gpus_curr, state_w, state_g, state_o, pop_w=False, pop_g=False, pop_o=False, pop_layer=None):
            gpus_new = create_gpus(state_w, state_g, state_o)
            self.play(FadeOut(gpus_curr, run_time=0.3), FadeIn(gpus_new, run_time=0.7))
            
            pop_rects = []
            for i in range(6):
                for comp_idx, (states, do_pop) in enumerate([(state_w[i], pop_w), (state_g[i], pop_g), (state_o[i], pop_o)]):
                    if do_pop:
                        for j in range(4):
                            if pop_layer is not None and j != pop_layer:
                                continue
                            if states[j] in [3, 4, 5]:
                                rect = gpus_new[i][1][1][comp_idx][j] 
                                pop_rects.append(rect)
                                
            if pop_rects:
                self.play(*[r.animate.scale(1.3) for r in pop_rects], run_time=0.2)
                self.play(*[r.animate.scale(1/1.3) for r in pop_rects], run_time=0.2)
                
            return gpus_new

        title_text = VGroup()
        desc_text = VGroup()
        title_bg = VGroup()
        is_first_time = True
        
        def update_center(new_title, new_desc):
            nonlocal title_text, title_bg, desc_text, is_first_time
            
            next_title = MarkupText(new_title, font_size=16, color=WHITE, weight=BOLD, font="Arial")
            next_desc = MarkupText(new_desc, font_size=12, color=LIGHT_GRAY, font="Arial")
            
            next_content = VGroup(next_title, next_desc).arrange(DOWN, buff=0.15).move_to(ORIGIN)
            next_bg = SurroundingRectangle(next_content, color=WHITE, stroke_width=1, fill_color="#1E3232", fill_opacity=1, corner_radius=0.1)
            
            if is_first_time:
                is_first_time = False
                title_text, desc_text, title_bg = next_title, next_desc, next_bg
                return [FadeIn(title_bg), FadeIn(title_text), FadeIn(desc_text)]
            else:
                anims = [
                    ReplacementTransform(title_bg, next_bg),
                    FadeOut(title_text, run_time=0.3), FadeIn(next_title, run_time=0.7),
                    FadeOut(desc_text, run_time=0.3), FadeIn(next_desc, run_time=0.7)
                ]
                title_text, desc_text, title_bg = next_title, next_desc, next_bg
                return anims

        header = MarkupText("ZeRO Stage 1:\nOptimizer State Partitioning", font_size=20, weight=BOLD, font="Arial").to_corner(UL)
        self.add(header)

        def make_state(val): return [[val]*4 for _ in range(6)]
        
        # 0=Empty, 1=Sharded Base, 2=Full Base, 3=Sharded Upd, 4=Full Upd, 5=Mixed Upd
        state_w = make_state(2) 
        state_g = make_state(0) 
        state_o = make_state(1) 

        # --- STEP 0: Init ---
        self.play(*update_center("Step 0: Initial State", "Weights replicated."))
        gpus_current = create_gpus(state_w, state_g, state_o)
        self.play(FadeIn(gpus_current))
        self.wait(2)

        # --- 1: Forward ---
        self.play(*update_center("Step 1: Forward Pass", "Compute activations with\nfull model weights."))
        self.wait(2)

        # --- 2: Backward ---
        self.play(*update_center("Step 2: Backward Pass", "Calculate full gradients\nfor each micro-batch."))
        state_g = make_state(2)
        gpus_current = animate_transition(gpus_current, state_w, state_g, state_o)
        self.wait(2)

        # --- 3: Average Gradient ---
        self.play(*update_center("Step 3: Average Gradient", "Reduce gradients."))
        state_g = make_state(4)
        gpus_current = animate_transition(gpus_current, state_w, state_g, state_o, pop_g=True)
        self.wait(2)

        # --- 4: Update Optimizer ---
        self.play(*update_center("Step 4: Update Optimizer", ""))
        state_o = make_state(3)
        gpus_current = animate_transition(gpus_current, state_w, state_g, state_o, pop_o=True)
        self.wait(2)

        # --- 5: Update Weight ---
        self.play(*update_center("Step 5: Update Weight", ""))
        state_w = make_state(5)
        state_g = make_state(0)
        gpus_current = animate_transition(gpus_current, state_w, state_g, state_o, pop_w=True)
        self.wait(2)

        # --- 6: All-Gather Weight ---
        self.play(*update_center("Step 6: All-Gather Weight", "Reconstruct full model."))
        state_w = make_state(4)
        gpus_current = animate_transition(gpus_current, state_w, state_g, state_o, pop_w=True)
        self.wait(3)