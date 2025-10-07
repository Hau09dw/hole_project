import torch

def decode_yolo_output_multi(det_outs, num_classes=1, img_size=640, device="cuda"):
    """
    det_outs: list of [B, (5+num_classes), H, W] (YOLO-style output for P3,P4,P5)
    return:
        batch_boxes: list of [N,4] (xyxy)
        batch_logits: list of [N,num_classes] (raw logits)
        batch_obj: list of [N,1] (raw logits for objectness)
    """
    batch_boxes, batch_logits, batch_obj = [], [], []

    for out in det_outs:  # loop over feature maps
        B, C, H, W = out.shape
        if C < 5:
            raise ValueError(f"Invalid YOLO output channels: {C}")


        out = out.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        out = out.view(B, -1, C)  # [B, H*W, C]

        for b in range(B):
            pred = out[b]  # [H*W, 5+num_classes]

            # box (cx,cy,w,h)
            cxcywh = pred[:, :4]
            boxes = cxcywh_to_xyxy(cxcywh, img_size)  # bạn có hàm convert, nếu chưa có thì mình sẽ viết thêm

            # objectness (raw logits)
            obj_logit = pred[:, 4:5]

            # class (raw logits)
            cls_logit = pred[:, 5:]

            batch_boxes.append(boxes.to(device))
            batch_obj.append(obj_logit.to(device))
            batch_logits.append(cls_logit.to(device))

    return batch_boxes, batch_logits, batch_obj


def cxcywh_to_xyxy(cxcywh, img_size=640):
    """
    cxcywh: [N,4] normalized (cx,cy,w,h) trong [0,1]
    return: [N,4] absolute xyxy
    """
    cx = cxcywh[:, 0] * img_size
    cy = cxcywh[:, 1] * img_size
    w = cxcywh[:, 2] * img_size
    h = cxcywh[:, 3] * img_size

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)
