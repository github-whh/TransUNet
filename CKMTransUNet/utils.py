import torch
import torch.nn.functional as F

def L2_loss(y_true, y_pred):
    cri = torch.nn.MSELoss()
    loss = cri(y_pred, y_true)
    return loss 

# def L1_loss(y_true, y_pred, scales=[1], eps=1e-3):
#     loss = 0
#     for scale in scales:
#         size = (int(y_true.shape[2] * scale), int(y_true.shape[3] * scale))
#         y_true_scaled = F.interpolate(y_true, size=size, mode='bilinear')
#         y_pred_scaled = F.interpolate(y_pred, size=size, mode='bilinear')
#         diff = torch.sqrt((y_true_scaled - y_pred_scaled)**2 + eps**2)
#         loss +=  diff.mean()
#     return loss / len(scales)

def Edge_loss(y_true, y_pred, scales=[1, 0.5]):
    loss = 0
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=y_true.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=y_true.device).view(1, 1, 3, 3)
    for scale in scales:
        size = int(y_true.shape[2] * scale), int(y_true.shape[3] * scale)
        y_true_scaled = F.interpolate(y_true, size=size, mode='bilinear')
        y_pred_scaled = F.interpolate(y_pred, size=size, mode='bilinear')
        edge_true = 0
        edge_pred = 0
        for c in range(y_true_scaled.shape[1]):  # 遍历所有通道
            edge_true_x = F.conv2d(y_true_scaled[:, c:c+1], kernel_x, padding=1)
            edge_true_y = F.conv2d(y_true_scaled[:, c:c+1], kernel_y, padding=1)
            edge_pred_x = F.conv2d(y_pred_scaled[:, c:c+1], kernel_x, padding=1)
            edge_pred_y = F.conv2d(y_pred_scaled[:, c:c+1], kernel_y, padding=1)
            edge_true += torch.sqrt(edge_true_x**2 + edge_true_y**2 + 1e-6)
            edge_pred += torch.sqrt(edge_pred_x**2 + edge_pred_y**2 + 1e-6)
        edge_true /= y_true_scaled.shape[1]
        edge_pred /= y_pred_scaled.shape[1]
        loss += F.l1_loss(edge_pred, edge_true)
    return loss / len(scales)


def Laplacian_loss(y_true, y_pred, levels=4, weights=[1.0, 0.5, 0.25, 0.125]):
    assert y_true.shape == y_pred.shape, "Input shapes must match"
    B, C, H, W = y_true.shape
    total_loss = 0.0
    for c in range(C):
        channel_loss = 0.0
        true_c = y_true[:, c:c+1]  # [B,1,H,W]
        pred_c = y_pred[:, c:c+1]  # [B,1,H,W]
        true_resized = true_c
        pred_resized = pred_c
        for l in range(levels):
            true_down = F.avg_pool2d(true_resized, 2)
            pred_down = F.avg_pool2d(pred_resized, 2)
            
            true_up = F.interpolate(true_down, scale_factor=2, 
                                  mode='bilinear', align_corners=False)
            pred_up = F.interpolate(pred_down, scale_factor=2,
                                  mode='bilinear', align_corners=False)
            h, w = true_resized.shape[-2:]
            true_lap = true_resized - true_up[:, :, :h, :w]
            pred_lap = pred_resized - pred_up[:, :, :h, :w]
            channel_loss += weights[l] * F.l1_loss(pred_lap, true_lap, reduction='mean')
            true_resized = true_down
            pred_resized = pred_down
        total_loss += channel_loss
    return total_loss / C

def calc_loss(pred, target, num, epoch, num_epochs):
    # if epoch < num_epochs // 2:
    lambda_L2, lambda_lap, lambda_edge = 1, 0.02, 0.01  # 固定权重
    # else:
    #     lambda_L2, lambda_lap, lambda_edge = 0.7, 0.2, 0.1  # 固定权
    L2_Loss = L2_loss(target, pred)
    Laplacian_Loss = Laplacian_loss(target, pred)
    Edge_Loss = Edge_loss(target, pred)
    Loss = lambda_L2 * L2_Loss + lambda_lap * Laplacian_Loss + lambda_edge * Edge_Loss
    if(num % 100 ==0):
        print(f"epoch: {epoch}, L2_loss: {L2_Loss.item():.4f}")
        # print(f"epoch: {epoch}, Laplacian_loss: {Laplacian_Loss.item():.4f}")
        # print(f"epoch: {epoch}, Edge_loss: {Edge_Loss.item():.4f}")
        # print(f"epoch: {epoch}, Loss: {Loss.item():.4f}")
    return Loss


