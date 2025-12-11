import numpy as np
import random
from game.board import Board

# ==========================================
# 1. CẤU HÌNH ĐIỂM SỐ (HEURISTIC)
# ==========================================
SCORES = {
    'WIN': 1000000,      # 5 ô liên tiếp: Thắng tuyệt đối.
    'OPEN_4': 50000,     # 4 ô mở 2 đầu (VD: .XXXX.): Chắc chắn thắng lượt sau (đối thủ chặn đầu này thì ta đánh đầu kia).
    'CLOSED_4': 10000,   # 4 ô bị chặn 1 đầu (VD: OXXXX.): Cần đánh ngay vào đầu còn lại nếu không muốn bị chặn hết.
    'OPEN_3': 5000,      # 3 ô mở 2 đầu (VD: .XXX.): Tiềm năng tạo thành Open 4.
    'CLOSED_3': 1000,    # 3 ô bị chặn 1 đầu: Ít nguy hiểm hơn Open 3.
    'OPEN_2': 500,       # 2 ô mở 2 đầu: Giai đoạn khai cuộc.
    'CLOSED_2': 100      # 2 ô bị chặn 1 đầu: Giá trị thấp nhất.
}

class MinimaxAgent:
    def __init__(self, player_id: int, depth: int = 2):
        """
        Khởi tạo Agent.
        - player_id: 1 (AI đi trước) hoặc 2 (AI đi sau).
        - depth: Độ sâu suy nghĩ. Depth=2 nghĩa là AI tính: "Mình đi -> Đối thủ đi -> Hết (Chấm điểm)".
        """
        self.player_id = player_id
        self.opponent_id = 3 - player_id  # Nếu mình là 1 thì đối thủ là 2 và ngược lại.
        self.depth = depth
        self.name = f"Minimax_D{depth}"

    def get_name(self):
        return self.name

    # ==========================================
    # 2. HÀM CHỌN NƯỚC ĐI (ROOT)
    # ==========================================
    def choose_move(self, board: Board):
        """Hàm này được gọi bởi main loop để AI quyết định nước đi."""
        best_score = -float('inf') # Khởi tạo điểm thấp nhất có thể
        best_move = None
        
        # Bước 1: Lấy danh sách candidates (local search)
        # Thay vì xét toàn bộ 225 ô, chỉ xét các ô gần quân cờ đã đánh.
        candidates = self.get_nearby_moves(board)
        
        # Bước 1.5: Trường hợp đặc biệt - bàn cờ trống
        # Trường hợp bàn cờ trống (lượt đầu tiên), luôn đánh vào giữa cho mạnh.
        if not candidates:
            center = board.size // 2
            return center, center

        # Bước 2: Sort candidates theo potential
        # Explore các move tốt trước -> Alpha-Beta cắt tỉa hiệu quả hơn
        candidates = self.order_moves(board, candidates)

        # Bước 3: Khởi tạo Alpha-Beta cho thuật toán cắt tỉa
        # Alpha: Điểm tốt nhất mà AI (Max) chắc chắn đạt được.
        # Beta: Điểm thấp nhất mà Đối thủ (Min) chắc chắn ép AI xuống được.
        alpha = -float('inf')
        beta = float('inf')

        # Bước 4: Duyệt qua từng nước đi tiềm năng
        for move in candidates:
            row, col = move
            
            # Bước 4.1: Thử đi (Simulate Move)
            board.grid[row][col] = self.player_id
            
            # Bước 4.2: Gọi đệ quy Minimax để xem tương lai nước đi này dẫn về đâu
            # Lưu ý: Truyền False vì lượt tiếp theo là của đối thủ (Minimizing)
            score = self.minimax(board, self.depth - 1, alpha, beta, False)
            
            # Bước 4.3: Hoàn tác (Undo/Backtrack) để trả lại bàn cờ sạch cho vòng lặp sau
            board.grid[row][col] = 0
            
            # Bước 4.4: Cập nhật nước đi tốt nhất
            if score > best_score:
                best_score = score
                best_move = move
            
            # Bước 4.5: Cập nhật Alpha (Ngưỡng dưới tốt nhất của AI)
            alpha = max(alpha, best_score)
            
        # Nếu có lỗi gì đó không tìm được move (hiếm gặp), chọn đại move đầu tiên
        return best_move if best_move else candidates[0]

    # ==========================================
    # 3. THUẬT TOÁN MINIMAX (CÓ CẮT TỈA)
    # ==========================================
    def minimax(self, board: Board, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """
        Hàm đệ quy cốt lõi.
        - is_maximizing = True: Lượt của AI (cố gắng lấy điểm cao nhất).
        - is_maximizing = False: Lượt của Đối thủ (cố gắng dìm điểm AI xuống thấp nhất).
        """
        
        # ĐIỀU KIỆN DỪNG (Base Case):
        # Nếu đã suy nghĩ hết độ sâu (depth=0), dừng lại và chấm điểm thế cờ hiện tại.
        if depth == 0:
            return self.evaluate_board(board, self.player_id)
        
        # Lấy danh sách các nước đi kế tiếp có thể xảy ra
        candidates = self.get_nearby_moves(board)
        
        # Nếu không còn nước nào để đi (Hòa) -> Điểm = 0
        if not candidates:
            return 0

        # --- PHẦN CỦA MAX (AI) ---
        if is_maximizing:
            max_eval = -float('inf')
            for move in candidates:
                r, c = move
                
                # Thử đi nước của AI
                board.grid[r][c] = self.player_id
                
                # Check Win sớm
                # Nếu nước này thắng luôn, không cần tính tiếp, trả về điểm MAX ngay.
                if self.check_win_quickly(board, r, c, self.player_id):
                    board.grid[r][c] = 0
                    return SCORES['WIN']
                
                # Đệ quy xuống tầng dưới (lượt đối thủ)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.grid[r][c] = 0 # Hoàn tác
                
                # Cập nhật giá trị Max
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                # [QUAN TRỌNG] Cắt tỉa Alpha-Beta
                # Nếu beta (giới hạn của đối thủ) <= alpha (giới hạn của mình)
                # Nghĩa là nhánh này đối thủ sẽ không bao giờ cho phép xảy ra -> Cắt bỏ, không tính nữa.
                if beta <= alpha:
                    break
            return max_eval
            
        # --- PHẦN CỦA MIN (ĐỐI THỦ) ---
        else:
            min_eval = float('inf')
            for move in candidates:
                r, c = move
                
                # Thử đi nước của Đối thủ
                board.grid[r][c] = self.opponent_id
                
                # Check Win sớm cho đối thủ
                if self.check_win_quickly(board, r, c, self.opponent_id):
                    board.grid[r][c] = 0
                    return -SCORES['WIN']   # GG

                # Đệ quy xuống tầng dưới (lượt AI: max)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.grid[r][c] = 0 # Hoàn tác
                
                # Cập nhật giá trị Min
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                # Cắt tỉa
                if beta <= alpha:
                    break
            return min_eval

    # ==========================================
    # 4. CÁC HÀM HỖ TRỢ & HEURISTIC
    # ==========================================
    def get_nearby_moves(self, board: Board, radius: int = 1):
        """
        Chiến lược 'Khoanh vùng': Chỉ quan tâm các ô trống nằm cạnh quân cờ đã có.
        - radius=1: Xét 8 ô xung quanh 1 quân cờ.
        Lý do: Trong Gomoku, hiếm khi ai đánh vào 1 ô xa tít mù tắp không liên quan.
        """
        played_points = np.argwhere(board.grid != 0)
        if len(played_points) == 0:
            return []
            
        candidates = set()
        size = board.size
        
        for r, c in played_points:
            # Xác định hình chữ nhật bao quanh quân cờ tại (r, c)
            r_min = max(0, r - radius)
            r_max = min(size, r + radius + 1)
            c_min = max(0, c - radius)
            c_max = min(size, c + radius + 1)
            
            # Duyệt các ô trong hình chữ nhật đó
            for i in range(r_min, r_max):
                for j in range(c_min, c_max):
                    if board.grid[i][j] == 0: # Chỉ lấy ô trống
                        candidates.add((i, j))
                        
        return list(candidates)

    def order_moves(self, board: Board, moves):
        """
        Sắp xếp các nước đi theo potential score (từ cao -> thấp).
        Mục đích: Alpha-Beta sẽ cắt tỉa tốt hơn nếu explore move tốt trước.
        """
        move_scores = []
        
        for move in moves:
            r, c = move
            
            # Simulate move để đánh giá nhanh
            board.grid[r][c] = self.player_id
            
            # Check xem move này có thắng ngay không
            if board.check_winner(r, c, self.player_id):
                board.grid[r][c] = 0
                # Thắng ngay -> Ưu tiên tuyệt đối
                return [move]
            
            # Check xem có chặn được đối thủ thắng không
            board.grid[r][c] = self.opponent_id
            blocks_win = board.check_winner(r, c, self.opponent_id)
            board.grid[r][c] = 0
            
            if blocks_win:
                # Chặn thắng cũng rất quan trọng
                move_scores.append((move, SCORES['WIN'] // 2))
            else:
                # Đánh giá nhanh dựa trên heuristic
                board.grid[r][c] = self.player_id
                quick_score = self.evaluate_board(board, self.player_id)
                board.grid[r][c] = 0
                move_scores.append((move, quick_score))
        
        # Sort theo điểm giảm dần
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, score in move_scores]

    def check_win_quickly(self, board: Board, row, col, player):
        """Wrapper để kiểm tra thắng thua nhanh."""
        return board.check_winner(row, col, player)

    def evaluate_board(self, board: Board, player_id: int) -> int:
        """
        Hàm chấm điểm thế cờ.
        Công thức: Điểm = (Tấn công của mình) - (Phòng thủ * 1.2)
        """
        my_score = self.evaluate_lines(board, player_id)
        opponent_score = self.evaluate_lines(board, 3 - player_id)
        
        # Nhân 1.2 vào điểm đối thủ để AI chơi an toàn hơn.
        # AI sẽ ưu tiên chặn đối thủ (vì điểm trừ cao) hơn là tự xây nước yếu của mình.
        return my_score - (opponent_score * 1.2)

    def evaluate_lines(self, board: Board, player: int) -> int:
        """Quét toàn bộ bàn cờ theo 4 hướng để tính điểm."""
        score = 0
        lines = []
        
        # 1. Lấy tất cả hàng ngang
        for r in range(board.size):
            lines.append(board.grid[r, :])
            
        # 2. Lấy tất cả hàng dọc
        for c in range(board.size):
            lines.append(board.grid[:, c])
            
        # 3. Lấy tất cả đường chéo (chỉ lấy đường chéo dài >= 5 ô)
        for offset in range(-board.size + 5, board.size - 4):
            lines.append(board.grid.diagonal(offset))               # Chéo chính (\)
            lines.append(np.fliplr(board.grid).diagonal(offset))    # Chéo phụ (/)
            
        # Tính điểm từng đường
        for line in lines:
            score += self.score_line(line, player)
            
        return score

    def score_line(self, line, player):
        """
        Phân tích patterns trên 1 đường thẳng bằng cách chuyển về chuỗi String.
        Cải tiến: Count TẤT CẢ patterns, không chỉ pattern đầu tiên.
        Ví dụ: Dòng "011101110" có 2 OPEN_3 -> tính cả 2.
        """
        s = "".join(map(str, line))
        p = str(player)
        opponent = str(3 - player)
        
        total = 0
        
        # Check Win trước - nếu có 5 con thì return ngay
        if p * 5 in s:
            return SCORES['WIN']
        
        # Count OPEN_4 (4 quân mở 2 đầu): .XXXX.
        # Pattern này cực kỳ nguy hiểm - thắng chắc lượt sau
        pattern = "0" + p*4 + "0"
        count = s.count(pattern)
        if count > 0:
            total += SCORES['OPEN_4'] * count
        
        # Count CLOSED_4 (4 quân bị chặn 1 đầu)
        # Có 2 dạng: OXXXX. hoặc .XXXXO
        # Dùng sliding window để tránh đếm trùng
        for i in range(len(s) - 4):
            segment = s[i:i+5]
            if segment.count(p) == 4 and segment.count('0') == 1:
                # Đảm bảo không phải Open 4 (đã count ở trên)
                if i > 0 and s[i-1] == '0' and s[i+4] == '0':
                    continue
                total += SCORES['CLOSED_4']
        
        # Count OPEN_3 (3 quân mở 2 đầu): .XXX.
        pattern = "0" + p*3 + "0"
        count = s.count(pattern)
        if count > 0:
            total += SCORES['OPEN_3'] * count
        
        # Count CLOSED_3 (3 quân bị chặn)
        # Sliding window để tìm các đoạn XXX với ít nhất 1 đầu bị chặn
        for i in range(len(s) - 2):
            if s[i:i+3] == p*3:
                # Check nếu không phải là phần của Open_3
                left_open = (i == 0 or s[i-1] != '0')
                right_open = (i+3 >= len(s) or s[i+3] != '0')
                
                # Nếu ít nhất 1 đầu bị chặn (không phải cả 2 đầu đều mở)
                if left_open or right_open:
                    # Kiểm tra không phải Open_3 (đã đếm rồi)
                    if not (i > 0 and i+3 < len(s) and s[i-1] == '0' and s[i+3] == '0'):
                        total += SCORES['CLOSED_3']
        
        # Count OPEN_2 (2 quân mở 2 đầu): .XX.
        pattern = "0" + p*2 + "0"
        count = s.count(pattern)
        if count > 0:
            total += SCORES['OPEN_2'] * count
        
        return total