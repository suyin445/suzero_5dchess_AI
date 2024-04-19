import abc


class Piece:
    @abc.abstractmethod
    def move_fuction(self, executability_validator):
        pass

    @abc.abstractmethod
    def inverse_capture_fuction(self, executability_validator):
        pass


class Pawn(Piece):
    def move_fuction(self, executability_validator):
        pass


class Rook(Piece):
    def move_fuction(self, executability_validator):
        pass


class Knight(Piece):
    def move_fuction(self, executability_validator):
        pass


class Bishop(Piece):
    def move_fuction(self, executability_validator):
        pass


class Queen(Piece):
    def move_fuction(self, executability_validator):
        pass


class King(Piece):
    def move_fuction(self, executability_validator):
        pass
