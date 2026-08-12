# -*- coding: utf-8 -*-
"""
gif_util.py  —  3D matplotlib 그림을 360° 회전 GIF 로 저장하는 공용 헬퍼.

규칙: like-slice 의 모든 3D 그림은 정지 PNG 대신 회전 GIF 로 만든다.

사용:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gif_util import save_rotate_gif
    ...
    fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
    # ax 에 scatter/quiver 등으로 그림
    save_rotate_gif(fig, ax, "out.gif")   # fig.savefig 대신
"""
import numpy as np
from PIL import Image


def save_rotate_gif(fig, ax, out_path, n_frames=36, elev=20,
                    duration=80, title=None):
    """이미 채워진 3D ax 를 azim 0→360 회전시키며 GIF 저장."""
    if title:
        ax.set_title(title)
    fig.tight_layout()
    frames = []
    for k in range(n_frames):
        ax.view_init(elev=elev, azim=k * (360.0 / n_frames))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    return out_path
