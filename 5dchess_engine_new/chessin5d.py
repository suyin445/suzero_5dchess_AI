import piece
import storage
import check_examine


class Chessin5d:
    def __init__(self, pieces = None, stalemate_examine = False):
        if pieces is None:
            self.pieces = default_pieces
        else:
            self.pieces = pieces
        self.available_actions_dic = {}
        self.moveable_pieces = None
        self.stalemate_examine = stalemate_examine
        self.check_examiner = check_examine.Check_examiner(self.pieces)

    def move_examine(self, move):
        if self.moveable_pieces is None:
            self.moveable_pieces = self.get_moveable_pieces()
        if self.moveable_pieces == [] and (not self.stalemate_examine):
            raise Exception("No moveable pieces")
        if move[0] not in self.moveable_pieces:
            return False
        if move[1] in self.available_actions_dic[move[0]]:
            return True
        the_piece = self.get_piece_type(move[0])
        for each in self.pieces[the_piece].move_fuction(self.check_examiner):
            self.available_actions_dic[move[0]].append(each)
        if move[1] in self.available_actions_dic[move[0]]:
            return True
        return False

    def new_timeline(self):11
        pass

    def get_moveable_pieces(self):
        pass

    def get_piece_type(self, position):
        pass


default_pieces = {1: piece.Pawn(), 2: piece.Rook(), 3: piece.Knight(), 4: piece.Bishop(), 5: piece.Queen(), 6: piece.King()}
