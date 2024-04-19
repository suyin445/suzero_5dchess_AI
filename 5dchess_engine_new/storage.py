import copy


class Board:
    def __init__(self, data=None):
        if data:
            self.board = data
        else:
            pass


class State:
    def __init__(self, turn):
        self.state = {}
        self.not_move = []
        self.present = 0
        self.turn = 1 if turn == 1 else 0
        self.whiteline = 0
        self.blackline = 0
        self.available_timeline = [0]
        pass

    def get_data_formatted(self):
        data = []
        for key, value in self.state.items():
            data.append({
                "id": key,
                "board": value
            })
        return data

    def copy(self):
        return copy.deepcopy(self)
