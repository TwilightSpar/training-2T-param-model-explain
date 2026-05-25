# %%manim -v WARNING -qm ZeROStage1
from manim import *
import numpy as np

class ZeROStage1(Scene):
    def construct(self):
        W_COLOR = "#5D9CEC"   
        W_UPDATED = "#2E86C1" 
        G_COLOR = "#D98880"   
        G_UPDATED = "#E74C3C" 
        O_COLOR = "#A8E6CF"   
        O_UPDATED = "#27AE60" 
        EMPTY = "#555555"     

        def create_gpus(state_w, state_g, state_o):
            gpus = VGroup()
            for i in range(6):
                blocks_group = VGroup()
                letters_group = VGroup()
                
                # Rows for Weights, Gradients, Optimizer (O gets taller height: 0.45 vs 0.25)
                for name, status, color, u_color, rect_h in [
                    ("W", state_w[i], W_COLOR, W_UPDATED, 0.25), 
                    ("G", state_g[i], G_COLOR, G_UPDATED, 0.25), 
                    ("O", state_o[i], O_COLOR, O_UPDATED, 0.45)
                ]:
                    letter = MarkupText(name, font_size=12, weight=BOLD, font="Helvetica Neue")
                    blocks = VGroup()
                    
                    for j in range(6):
                        sq = Rectangle(height=rect_h, width=0.35, stroke_width=1, stroke_color=WHITE)
                        if status[j] == 1: sq.set_fill(color, 1.0)
                        elif status[j] == 2: sq.set_fill(u_color, 1.0) 
                        else: sq.set_fill(EMPTY, 1.0) 
                        
                        num = MarkupText(str(j), font_size=10, color=WHITE, font="Helvetica Neue").move_to(sq.get_center())
                        blocks.add(VGroup(sq, num))
                    
                    blocks.arrange(RIGHT, buff=0)
                    blocks_group.add(blocks)
                    letters_group.add(letter)
                
                blocks_group.arrange(DOWN, buff=0.1)
                for letter, blocks in zip(letters_group, blocks_group):
                    letter.next_to(blocks, LEFT, buff=0.1)
                
                gpu_content = VGroup(letters_group, blocks_group)
                box = SurroundingRectangle(gpu_content, buff=0.1, color=WHITE, stroke_width=1, corner_radius=0.05)
                title = MarkupText(f"GPU {i}", font_size=14, weight=BOLD, font="Helvetica Neue").next_to(box, DOWN, buff=0.1)
                
                node = VGroup(box, gpu_content, title)
                angle = np.pi/2 - i * (2 * np.pi / 6)
                node.move_to(DOWN*0.2 + np.array([3.1 * np.cos(angle), 3.1 * np.sin(angle), 0]))
                gpus.add(node)
                
            return gpus

        def animate_transition(gpus_curr, state_w, state_g, state_o, pop_w=False, pop_g=False, pop_o=False):
            gpus_new = create_gpus(state_w, state_g, state_o)
            self.play(FadeOut(gpus_curr, run_time=0.5), FadeIn(gpus_new, run_time=1.0))
            
            pop_rects = []
            for i in range(6):
                for layer, states, do_pop in [(0, state_w[i], pop_w), (1, state_g[i], pop_g), (2, state_o[i], pop_o)]:
                    if do_pop:
                        for j in range(6):
                            if states[j] == 2:
                                rect = gpus_new[i][1][1][layer][j][0]
                                pop_rects.append(rect)
            
            # Pulse effect: scale up then immediately back to normal
            if pop_rects:
                self.play(*[r.animate.scale(1.4) for r in pop_rects], run_time=0.2)
                self.play(*[r.animate.scale(1/1.4) for r in pop_rects], run_time=0.2)
                
            return gpus_new

        title_text = VGroup()
        desc_text = VGroup()
        title_bg = VGroup()
        is_first_time = True
        
        def update_center(new_title, new_desc):
            nonlocal title_text, title_bg, desc_text, is_first_time
            
            next_title = MarkupText(new_title, font_size=16, color=WHITE, weight=BOLD, font="Helvetica Neue")
            next_desc = MarkupText(new_desc, font_size=12, color=LIGHT_GRAY, font="Helvetica Neue")
            
            next_content = VGroup(next_title, next_desc).arrange(DOWN, buff=0.15).move_to(ORIGIN)
            next_bg = SurroundingRectangle(next_content, color=WHITE, stroke_width=1, fill_color="#1E3232", fill_opacity=1, corner_radius=0.1)
            
            if is_first_time:
                is_first_time = False
                title_text, desc_text, title_bg = next_title, next_desc, next_bg
                return [FadeIn(title_bg), FadeIn(title_text), FadeIn(desc_text)]
            else:
                anims = [
                    ReplacementTransform(title_bg, next_bg),
                    FadeOut(title_text, run_time=0.5), FadeIn(next_title, run_time=1.0),
                    FadeOut(desc_text, run_time=0.5), FadeIn(next_desc, run_time=1.0)
                ]
                title_text, desc_text, title_bg = next_title, next_desc, next_bg
                return anims

        # Header is now smaller and uses a line break to avoid overlap
        header = MarkupText("ZeRO Stage 1:\nOptimizer State Partitioning", font_size=25, weight=BOLD, font="Helvetica Neue").to_corner(UL)
        self.add(header)

        full_1 = [[1]*6 for _ in range(6)]
        full_2 = [[2]*6 for _ in range(6)]
        empty = [[0]*6 for _ in range(6)]
        diag_1 = [[1 if i==j else 0 for j in range(6)] for i in range(6)]
        diag_2 = [[2 if i==j else 0 for j in range(6)] for i in range(6)]

        # --- STEP 0 ---
        self.play(*update_center("Initial State", "Optimizer sharded.\nWeights replicated."))
        gpus_current = create_gpus(full_1, empty, diag_1)
        self.play(FadeIn(gpus_current))
        self.wait(2)

        # --- STEP 1 ---
        self.play(*update_center("Fwd and Bwd Pass", "Calculate full gradients\nfor micro-batch."))
        gpus_current = animate_transition(gpus_current, full_1, full_1, diag_1)
        self.wait(2)

        # --- STEP 2 ---
        self.play(*update_center("All-Reduce Gradients", "Average gradients.\nAll GPUs hold full copy."))
        gpus_current = animate_transition(gpus_current, full_1, full_2, diag_1, pop_g=True)
        self.wait(2)

        # --- STEP 3 ---
        self.play(*update_center("Weight Update", "Use Optimizer chunk to update.\nDiscard unused Gradients."))
        w_step3 = [[2 if i==j else 1 for j in range(6)] for i in range(6)]
        gpus_current = animate_transition(gpus_current, w_step3, empty, diag_2, pop_w=True, pop_o=True)
        self.wait(2)

        # --- STEP 4 ---
        self.play(*update_center("All-Gather Weights", "Broadcast updated weights\nto full model."))
        gpus_current = animate_transition(gpus_current, full_2, empty, diag_1, pop_w=True)
        self.wait(3)