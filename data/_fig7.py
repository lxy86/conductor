import pymupdf
doc = pymupdf.open("miehe2010.pdf")
page = doc[8]
for blk in page.get_image_info(xrefs=True):
    if blk["xref"] == 87:
        r = pymupdf.Rect(blk["bbox"])
        # 只取右半张图 (b) 的左下部分：线性段
        left = pymupdf.Rect(r.x0 + 0.52*r.width, r.y0 + 0.30*r.height,
                            r.x0 + 0.85*r.width, r.y1)
        page.get_pixmap(dpi=900, clip=left).save("data/_fig7b_linear.png")
        print("saved")
