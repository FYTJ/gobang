# 人工智能：现代方法 P126
# chatgpt title: Minimax搜索算法、五子棋AI优化建议
import datetime
import random
import re
import cProfile
import time

import numpy as np


class Game:
    """一个游戏对象，封装了游戏规则、状态转移、效用计算逻辑"""

    def __init__(self):
        self.board = np.array([[0 for x in range(15)] for y in range(15)])
        self.current_player = 1  # 黑先
        self.record_chess = []
        self.ban_patterns = {
            'three_patterns': ['011100', '001110', '010110', '011010'],
            'non_three_patterns': ['20111001', '10011102'],
            'four_patterns': ['011110', '11101', '10111', '11011'],
            'single_patterns': ['1011101', '10111101', '111010111', '11011011']
        }

    def to_move(self) -> int:
        """
        返回当前轮到行动的玩家
        黑为1，白为2
        """
        return self.current_player

    def display_board(self, last_mov=None) -> None:
        mov_x, mov_y = None, None
        if last_mov is not None:
            mov_x, mov_y = last_mov
        for x in range(14, -1, -1):
            line = [f'{x + 1:<2}'] + [cell for cell in self.board[x]].copy()
            if x == mov_x:
                line[1 + mov_y] += 2  # 行号占1位
            print(' '.join([f'{i:<2}' for i in line]))
        print(' '.join([f'{i:<2}' for i in range(0, 16)]))
        print('\n')

    def play(self, move: (int, int)) -> bool:
        if move is not None:
            line, column = move
            if self.board[line][column]:
                return False
            self.board[line][column] = self.current_player
            self.current_player = 3 - self.current_player
            self.record_chess.append((line, column))
            return True

    def is_cutoff(self, state: np.ndarray, depth: int) -> bool:
        """
        判断当前状态是否为终止状态
        :note: 由于存在单步延伸，需要考虑depth < 0的情况
        """
        return self.check_winner(state) is not None or depth <= 0

    @staticmethod
    def actions(state: np.ndarray):
        """返回当前状态下所有合法的行动"""
        return [(x, y) for x in range(15) for y in range(15) if state[x][y] == 0]

    @staticmethod
    def neighbors(state: np.ndarray):
        """返回当前局面的邻域：向外延伸两个点"""
        neighbor = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, -1), (-1, 1)]
        for x in range(15):
            for y in range(15):
                add = 0
                for dx, dy in directions:
                    for r in (1, 2):
                        # if 0 <= x + r * dx < 15 and 0 <= y + r * dy < 15 and state[x + dx][y + dy] and not state[x][y]:
                        # 2024/12/08 22:51 neighbor.append((x + r * dx, y + r * dy))
                        if 0 <= x + r * dx < 15 and 0 <= y + r * dy < 15 and state[x + r * dx, y + r * dy] and not \
                                state[x][y]:
                            add = 1
                if add:
                    neighbor.append((x, y))
        return neighbor

    @staticmethod
    def result(state: np.ndarray, action: (int, int), player) -> np.ndarray:
        """返回执行某一行动后的新状态"""
        x, y = action
        new_state = state.copy()
        new_state[x][y] = player
        return new_state

    def check_winner(self, state: np.ndarray) -> int | None:
        """
        裁判
        0: dual; 1: black; 2: white
        """
        if self.check_ban(state, 1, self.record_chess[-1]):
            return 2
        for x in range(15):
            for y in range(15):
                for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    line = ''
                    for step in range(-4, 5):
                        current_x, current_y = x + step * dx, y + step * dy
                        if 0 <= current_x < 15 and 0 <= current_y < 15:
                            line += str(state[current_x][current_y])
                    if '11111' in line:
                        return 1
                    if '22222' in line:
                        return 2
        for cell in state.flatten():
            if cell == 0:
                return None
        return 0

    def check_ban(self, state: np.ndarray, player: int, last_mov: (int, int)) -> bool:
        """
        检查禁手
        """
        if self._long_six(state, player, last_mov):
            return True
        if player != 1 and self._double_three(state, player, last_mov) or self._double_four(state, player, last_mov):
            return True
        return False

    def _double_three(self, state: np.ndarray, player: int, last_mov: (int, int)) -> bool:
        counter = 0
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            x, y = last_mov
            line = ''
            for step in range(-4, 5):
                current_x, current_y = x + step * dx, y + step * dy
                if 0 <= current_x < 15 and 0 <= current_y < 15:
                    content = state[current_x][current_y]
                    if content == player:
                        line += '1'
                    elif content == 0:
                        line += '0'
                    else:
                        line += '2'
                else:
                    line += '2'  # 边界当作对方子处理
            if self._list_in_string(self.ban_patterns['three_patterns'], line) \
                    and not self._list_in_string(self.ban_patterns['four_patterns'], line) \
                    and not self._list_in_string(self.ban_patterns['non_three_patterns'], line):
                counter += 1
            if counter == 2:
                return True
        return False

    def _double_four(self, state: np.ndarray, player: int, last_mov: (int, int)) -> bool:
        counter = 0
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            x, y = last_mov
            line = ''
            for step in range(-3, 4):
                current_x, current_y = x + step * dx, y + step * dy
                if 0 <= current_x < 15 and 0 <= current_y < 15:
                    content = state[current_x][current_y]
                    if content == player:
                        line += '1'
                    elif content == 0:
                        line += '0'
                    else:
                        line += '2'
                else:
                    line += '2'  # 边界当作对方子处理
            if self._list_in_string(self.ban_patterns['four_patterns'], line):
                counter += 1
            if self._list_in_string(self.ban_patterns['single_patterns'], line):
                return True
            if counter == 2:
                return True
        return False

    def _long_six(self, state: np.ndarray, player: int, last_mov: (int, int)) -> bool:
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            x, y = last_mov
            line = ''
            for step in range(-3, 4):
                current_x, current_y = x + step * dx, y + step * dy
                if 0 <= current_x < 15 and 0 <= current_y < 15:
                    content = state[current_x][current_y]
                    if content == player:
                        line += '1'
                    elif content == 0:
                        line += '0'
                    else:
                        line += '2'
                else:
                    line += '2'  # 边界当作对方子处理
            if self._list_in_string(['111111'], line):
                return True
        return False

    @staticmethod
    def _list_in_string(l: list, string: str) -> bool:
        for item in l:
            if item in string:
                return True
        return False


class AI:
    def __init__(self, name: str, color: int):
        self.count = None
        self.maxv = None
        self.minv = None
        self.name = name
        self.color = color  # 黑1白2
        self.lmr_threshold = 10
        self.lmr_min_depth = 0
        self.patterns = [
            (r'11111', 50000),  # 连五
            (r'011110', 4320),  # 活四
            (r'011112|211110', 720),  # 冲四
            (r'011100|001110|011010|010110|11011|10111|11101', 720),  # 活三
            (r'001100|001010|010100', 120),  # 活二
            (r'000112|211000|001012|210100|010012|210010|10001', 50),  # 眠二
            (r'000100|001000', 20),
        ]
        self.openings = []
        self.read_openings()

    def __repr__(self):
        return f'{self.name}'

    def read_openings(self):
        """
        从文件读取开局表
        """
        with (open('./gobang/openings.txt', 'r') as f):
            lines = f.readlines()
            for line in lines:
                line = line.replace('[', '').replace(']', '').replace('\n', ''
                                                                      ).replace(' ', '').replace('(', '').replace(')',
                                                                                                                  '').split(
                    ',')
                self.openings.append([(int(line[2 * i]), int(line[2 * i + 1])) for i in range(5)])

    def look_up_table(self, record_chess: [(int, int)]):
        """
        查开局表
        :param record_chess: ([(int, int)]) 棋盘
        :return: (int, int) | None下一步移动
        """
        steps = len(record_chess)
        random.seed = datetime.datetime.now()
        random.shuffle(self.openings)
        for opening in self.openings:
            if opening[: steps] == record_chess:
                return opening[steps]
        return None

    def heuristic_alpha_beta_search(self, game: Game, depth: int, is_first_move) -> (int, int):
        """
        极小化极大计算最优移动，启发式alpha-beta剪枝。
        结合概率截断和后期移动缩减(LMR)。概率截断适合空盘和中盘阶段，LMR适合后期关键局面。
        对排序靠后的动作，根据概率决定是否忽略，对于未被截断的动作，减少搜索深度。
        :param game: (Game) 游戏对象
        :param depth: (int) 搜索深度
        :param is_first_move: (bool) 是否是第一个子
        :return: (int, int) 落子位置
        """
        self.count, self.maxv, self.minv = 0, 0, 0
        move = None
        if is_first_move:
            return 7, 7
        if len(game.record_chess) <= 4:
            move = self.look_up_table(game.record_chess)
        if move is None:
            _, move = self._max_value(game, game.board, self.color, float('-inf'), float('inf'), depth)
        print(self.count, self.maxv, self.minv)
        return move

    def _max_value(self, game: Game, state: np.ndarray, player: int, alpha: int | float, beta: int | float,
                   depth: int) -> (int, (int, int)):
        """
        计算最大值的辅助函数
        :param game: (Game) 游戏对象
        :param state: (np.ndarray) 棋盘
        :param player: (int) 当前玩家
        :param alpha: (int | float) 当前最大下界
        :param beta: (int | float) 当前最小上界
        :param depth: (int) 搜索深度
        :return: (int, (int, int)) (效用值, 最佳行动)
        :var self.lmr_threshold: (int) 后期移动缩减动作序数阈值，该值是depth的函数
        :var self.lmr_min_depth: (int) 在达到一定深度后进行LMR
        """
        v, move = float('-inf'), None
        self.lmr_threshold = 10
        self.lmr_min_depth = 3
        current_eval = self.heuristic_eval(game, state, player)
        if game.is_cutoff(state, depth):
            return current_eval, None
        act = {}
        for next_pos in game.actions(state):
            act[next_pos] = self.heuristic_eval(game, game.result(state, next_pos, player),
                                                player, next_pos, current_eval, state)
        act = dict(sorted(act.items(), key=lambda i: i[1], reverse=True))
        actions = list(act.keys())
        self.maxv += 1
        for order, pos in enumerate(actions):
            if pos not in game.neighbors(state):
                continue
            else:
                # LMR
                if order >= self.lmr_threshold and depth >= self.lmr_min_depth:
                    continue
                v_new, _ = self._min_value(game, game.result(state, pos, player), player, alpha, beta, depth - 1)
            if v_new > v:
                v, move = v_new, pos
            if v >= beta:
                # 剪枝
                return v, move
            alpha = max(alpha, v)
        return v, move

    def _min_value(self, game: Game, state: np.ndarray, player: int, alpha: int | float, beta: int | float,
                   depth: int) -> (int, (int, int)):
        """
        计算最小值的辅助函数
        :param game: (Game) 游戏对象
        :param state: (np.ndarray) 棋盘
        :param player: (int) 当前玩家
        :param alpha: (int | float) 当前最大下界
        :param beta: (int | float) 当前最小上界
        :param depth: (int) 搜索深度
        :return: (int, (int, int)) (效用值, 最佳行动)
        :var self.lmr_threshold: (int) 后期移动缩减动作序数阈值，该值是depth的函数
        :var self.lmr_min_depth: (int) 在达到一定深度后进行LMR
        """
        v, move = float('inf'), None
        self.lmr_threshold = 10
        self.lmr_min_depth = 2
        current_eval = self.heuristic_eval(game, state, player)
        if game.is_cutoff(state, depth):
            return current_eval, None
        act = {}
        for next_pos in game.actions(state):
            act[next_pos] = self.heuristic_eval(game, game.result(state, next_pos, 3 - player),
                                                player, next_pos, current_eval, state)
        act = dict(sorted(act.items(), key=lambda i: i[1]))
        actions = list(act.keys())
        self.minv += 1
        for order, pos in enumerate(actions):
            if pos not in game.neighbors(state):
                continue
            else:
                # LMR
                if order >= self.lmr_threshold and depth >= self.lmr_min_depth:
                    continue
                v_new, _ = self._max_value(game, game.result(state, pos, 3 - player), player, alpha, beta, depth - 1)
            if v_new < v:
                v, move = v_new, pos
            if v <= alpha:
                # 剪枝
                return v, move
            beta = min(beta, v)
        return v, move

    def heuristic_eval(self, game: Game, state: np.ndarray, player: int, last_mov: (int, int) = None,
                       prev_eval: int = None, prev_state: np.ndarray = None) -> int:
        """
        计算当前状态对指定玩家的评价
        :param game: (Game) 游戏对象
        :param state: (np.ndarray) 棋盘
        :param player: (int) 当前玩家
        :param last_mov: (int, int) 上一次落子位置，只需要评估该位置附近的区域。
        :param prev_eval: (int) 上一次的评估得分。如果为None，则为第一次评估，需要对整个棋盘进行评估。
        :param prev_state: (np.ndarray) 未在last_mov处落子时的棋盘，默认值为None，用于增量更新
        :return: (int) 评估得分
        :note: 增量更新条件：last_mov, prev_eval, prev_state 同时有值
        :note: 增量更新逻辑如下
            current_eval = prev_eval + target_eval + \delta_eval - \delta_opponent_eval
            \delta_eval, \delta_opponent 初始化为0
            target_eval针对last_mov处的子进行评价
            确定last_mov的作用域为：以last_mov为中心四个方向range(-4, 5)\{0}的棋盘点
            对作用域上的一个方向direction上的一个子:
            \delta_target = _eval_pos(state, player, target_pos, direction)
             - _eval_pos(prev_state, player, target_pos, direction)
             若为player，则\delta_eval += \delta_target，3 - player亦然
        """
        self.count += 1
        if last_mov and game.check_ban(state, player, last_mov):
            return -250000
        if prev_eval and last_mov and prev_state is not None:
            # 增量更新
            all_directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
            target_eval = delta_eval = delta_opponent_eval = 0
            for direction in all_directions:
                target_eval += self._eval_pos(state, player, last_mov, direction)
            for direction in all_directions:
                for step in list(range(-4, 0)) + list(range(1, 5)):
                    x = last_mov[0] + step * direction[0]
                    y = last_mov[1] + step * direction[1]
                    if 0 <= x < 15 and 0 <= y < 15:
                        if state[x][y] == player:
                            current_target_eval = self._eval_pos(state, player, (x, y), direction)
                            prev_target_eval = self._eval_pos(prev_state, player, (x, y), direction)
                            delta_eval += current_target_eval - prev_target_eval
                        elif state[x][y] == 3 - player:
                            current_target_eval = self._eval_pos(state, 3 - player, (x, y), direction)
                            prev_target_eval = self._eval_pos(prev_state, 3 - player, (x, y), direction)
                            delta_opponent_eval += current_target_eval - prev_target_eval
            return prev_eval + target_eval + delta_eval - delta_opponent_eval * 1.2
        else:
            # 全盘评估
            total_eval = 0
            directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
            for x in range(15):
                for y in range(15):
                    for direction in directions:
                        if state[x][y] == player:
                            total_eval += self._eval_pos(state, player, (x, y), direction)
                        elif state[x][y] == 3 - player:
                            total_eval -= self._eval_pos(state, 3 - player, (x, y), direction) * 1.2
            return total_eval

    def _eval_pos(self, state: np.ndarray, player: int, pos: (int, int), direction: (int, int)) -> int:
        """
        对指定玩家和指定位置的棋子计算评价
        :param state: (np.ndarray) 棋盘
        :param player: (int) 玩家
        :param pos: (int, int) 目标评价位置
        :param direction: [(int, int)] 方向
        """
        x, y = pos
        total_eval = 0
        dx, dy = direction
        line = self._get_line(state, player, x, y, dx, dy)
        total_eval += self._match_patterns(line)
        return total_eval

    @staticmethod
    def _get_line(state: np.ndarray, player: int, x: int, y: int, dx: int, dy: int) -> str:
        """
        获取一个方向上的棋盘
        :param state: (np.ndarray) 棋盘
        :param player: (int) 玩家
        :param x: (int) 当前棋子的x坐标
        :param y: (int) 当前棋子的y坐标
        :param dx: (int) x的变化步长
        :param dy: (int) y的变化步长
        :return: (str) dx, dy方向上的棋盘
        """
        line = ''
        # 搜索范围-4 ~ 4
        for step in range(-4, 5):
            current_x, current_y = x + step * dx, y + step * dy
            if 0 <= current_x < 15 and 0 <= current_y < 15:
                content = state[current_x][current_y]
                if content == player:
                    line += '1'
                elif content == 0:
                    line += '0'
                else:
                    line += '2'
            else:
                line += '2'  # 边界当作对方子处理
        return line

    def _match_patterns(self, line: str) -> int:
        score = 0
        for pattern, value in self.patterns:
            matches = re.findall(pattern, line)
            if matches:
                score += len(matches) * value
        return score


def human_play(game: Game):
    line, column = [int(i) for i in input('your move (line, column): ').split()]
    line -= 1
    column -= 1
    while not game.play((line, column)):
        print('try again')
        x, y = [int(i) for i in input().split()]
        x -= 1
        y -= 1
    game.display_board((line, column))


def ai_play(game: Game, ai: AI, is_first_move=False):
    start = time.time()
    move = ai.heuristic_alpha_beta_search(game, 3, is_first_move)
    end = time.time()
    game.play(move)
    print(ai.__repr__(), move[0] + 1, move[1] + 1)
    print('time cost:', end - start)
    game.display_board(move)


def main():
    game = Game()
    mode = int(input('1: human-human; 2: human-ai; 3: ai-ai\nchoose: '))
    if mode == 1:
        game.display_board()
        while game.check_winner(game.board) is None:
            human_play(game)
            if game.check_winner(game.board):
                break
            human_play(game)
        print(game.record_chess)
    elif mode == 2:
        human_color = int(input('Choose black or white.\n1: black; 2: white\nchoose: '))
        ai = AI('ai', 2 - human_color)
        game.display_board()
        if human_color == 2:
            ai_play(game, ai, is_first_move=True)
        while game.check_winner(game.board) is None:
            human_play(game)
            if game.check_winner(game.board):
                break
            ai_play(game, ai)
        print(game.record_chess)
    elif mode == 3:
        ai1 = AI('ai1', 1)
        ai2 = AI('ai2', 2)
        game.display_board()
        ai_play(game, ai1, is_first_move=True)
        while game.check_winner(game.board) is None:
            ai_play(game, ai2)
            if game.check_winner(game.board):
                break
            ai_play(game, ai1)
        print(game.record_chess)
    elif mode == 4:
        existed_chess = [
                         ]
        for pos in existed_chess:
            game.play(pos)
        ai1 = AI('ai1', 1)
        ai2 = AI('ai2', 2)
        game.display_board(existed_chess[-1])
        ai_play(game, ai1)


if __name__ == '__main__':
    main()
    # cProfile.run('main()')
