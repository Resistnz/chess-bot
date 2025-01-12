import pygame
from pygame import gfxdraw
from enum import Enum
import time

pygame.init()

# Setup the screen
win = pygame.display.set_mode((360, 360))
clock = pygame.time.Clock()

pygame.display.set_caption("Chess")

icon = pygame.image.load(f"pieces-png/bK.png")
pygame.display.set_icon(icon)

# region Datatypes
class PieceType(Enum):
    BISHOP = 0
    KING = 1
    KNIGHT = 2
    PAWN = 3
    QUEEN = 4
    ROOK = 5

class MoveType(Enum):
    SHORT_CASTLE = 0,
    LONG_CASTLE = 1,
    QUEEN_PROMOTION = 2,
    ROOK_PROMOTION = 3,
    KNIGHT_PROMOTION = 4,
    BISHOP_PROMOTION = 5,
    EN_PASSANT = 6,
    NORMAL = 7

promotionMoveTypes = [MoveType.QUEEN_PROMOTION, MoveType.BISHOP_PROMOTION, MoveType.KNIGHT_PROMOTION, MoveType.ROOK_PROMOTION]

Colour = tuple[int, int, int]
Square = tuple[int, int]

letterToPiece = {
    "r": PieceType.ROOK,
    "n": PieceType.KNIGHT,
    "b": PieceType.BISHOP,
    "k": PieceType.KING,
    "q": PieceType.QUEEN,
    "p": PieceType.PAWN
}

pieceValues = {
    PieceType.ROOK: 5,
    PieceType.KNIGHT: 3,
    PieceType.BISHOP: 3,
    PieceType.KING: 999,
    PieceType.QUEEN: 9,
    PieceType.PAWN: 1
}

pieceToLetter = {value: key for key, value in letterToPiece.items()}

def SquareToGridPos(square: Square) -> int:
    gridPos = square[1] * 8 + square[0]

    if not 0 <= gridPos < 64: return -1
    return gridPos

def SquareToNotation(square: Square):
    letters = 'abcdefgh'

    return f'{letters[square[0]]}{8-square[1]}'

def AddSquares(square1: Square, square2: Square) -> Square:
    return (square1[0] + square2[0], square1[1] + square2[1])

class Board():
    def __init__(self):
        self.board = [None] * 64
        self.pieces = []
        self.moveHistory = []

        self.isWhiteTurn = True
        self.whiteKingInCheck = False
        self.blackKingInCheck = False

        self.checkmate = False

    def AddPiece(self, piece: PieceType, square: Square) -> None:
        self.board[SquareToGridPos(square)] = piece

    def GetPieceAtSquare(self, square: Square):
        g = SquareToGridPos(square)

        if g == -1:
            return None

        return self.board[g]
    
    def MakeMove(self, move) -> None:
        piece = move.piece
        targetSquare = move.targetSquare
        capturedPiece = move.capturedPiece
        moveType = move.moveType

        self.board[SquareToGridPos(piece.square)] = None

        piece.MakeMove(move)

        #print(f"Making move: {move.ToNotation()}")

        # Promotions
        if moveType == MoveType.QUEEN_PROMOTION: piece.Promote(PieceType.QUEEN)
        if moveType == MoveType.ROOK_PROMOTION: piece.Promote(PieceType.ROOK)
        if moveType == MoveType.BISHOP_PROMOTION: piece.Promote(PieceType.BISHOP)
        if moveType == MoveType.KNIGHT_PROMOTION: piece.Promote(PieceType.KNIGHT)

        # Handle special move types
        if moveType == MoveType.SHORT_CASTLE and piece.pieceType == PieceType.KING:
            board.MakeMove(Move(capturedPiece, capturedPiece.square, AddSquares(targetSquare, (-1, 0)), None, MoveType.SHORT_CASTLE))

        elif moveType == MoveType.LONG_CASTLE and piece.pieceType == PieceType.KING:
            board.MakeMove(Move(capturedPiece, capturedPiece.square, AddSquares(targetSquare, (1, 0)), None, MoveType.LONG_CASTLE))

        # Capture whatever was there
        else:
            if capturedPiece != None:
                self.board[SquareToGridPos(capturedPiece.square)] = None

                capturedPiece.Capture()

        self.board[SquareToGridPos(targetSquare)] = piece

        # Avoid changing turn twice when castling
        if not (piece.pieceType == PieceType.ROOK and (moveType == MoveType.SHORT_CASTLE or moveType == MoveType.LONG_CASTLE)):
            self.isWhiteTurn = not self.isWhiteTurn

            self.moveHistory.append(move)

    def UnmakeLastMove(self) -> None:
        lastMove: Move = self.moveHistory.pop()

        piece = lastMove.piece
        targetSquare = lastMove.targetSquare
        capturedPiece = lastMove.capturedPiece
        moveType = lastMove.moveType

        reverseMove = Move(piece, targetSquare, lastMove.startSquare, capturedPiece, moveType)

        # Unpremote pieces
        if moveType in promotionMoveTypes:
            piece.Promote(PieceType.PAWN)

        #print(f"Unmaking move: {lastMove.ToNotation()}\n")

        # Undo castling
        if moveType == MoveType.SHORT_CASTLE and piece.pieceType == PieceType.KING:
            self.board[SquareToGridPos(capturedPiece.square)] = None

            rookTarget = AddSquares(targetSquare, (1, 0))
            reverseRookMove = Move(capturedPiece, AddSquares(targetSquare, (-1, 0)), rookTarget, None, MoveType.SHORT_CASTLE)

            # Fix the board
            self.board[SquareToGridPos(rookTarget)] = capturedPiece
            self.board[SquareToGridPos(targetSquare)] = None

            capturedPiece.MakeMove(reverseRookMove, unmake = True)

        elif moveType == MoveType.LONG_CASTLE and piece.pieceType == PieceType.KING:
            self.board[SquareToGridPos(capturedPiece.square)] = None

            rookTarget = AddSquares(targetSquare, (-2, 0))
            reverseRookMove = Move(capturedPiece, AddSquares(targetSquare, (1, 0)), rookTarget, None, MoveType.LONG_CASTLE)

            # Fix the board
            self.board[SquareToGridPos(rookTarget)] = capturedPiece
            self.board[SquareToGridPos(targetSquare)] = None

            capturedPiece.MakeMove(reverseRookMove, unmake = True)
        
        # Swap the pieces
        else:
            self.board[SquareToGridPos(targetSquare)] = capturedPiece

            if reverseMove.capturedPiece != None:
                reverseMove.capturedPiece.isCaptured = False
                reverseMove.capturedPiece.square = lastMove.targetSquare

        # Apply the reversed move
        self.board[SquareToGridPos(lastMove.startSquare)] = piece

        lastMove.piece.MakeMove(reverseMove, unmake = True)

        self.isWhiteTurn = not self.isWhiteTurn
# endregion

# region Images
pieceImages = []
pieceNames = ['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']

for p in pieceNames:
    image = pygame.image.load(f"pieces-png/{p}.png")

    pieceImages.append(image)
# endregion

# region UI
class Element():
    def __init__(self, x: int, y: int, w: int, h: int, col: Colour = (255, 255, 255), text: str = None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.text = text

        # Keep track of different hover/held colours
        self.defaultColour = col
        self.currentColour = col

    def Draw(self):
        pygame.draw.rect(win, self.currentColour, (self.x, self.y, self.w, self.h))

        #if self.text != None:
        #    t = letterFont.render(self.text, True, (255, 255, 255))
        #    win.blit(t, (self.x + self.w/2 - letterFont.size(self.text)[0]/2, self.y + self.h/2 - letterFont.size(self.text)[1]/2))

    # Multiply the colour tuple by some amount
    def _darken(self, colToDarken: Colour, darkenAmount: float) -> Colour:
        return tuple(int(c * darkenAmount) for c in colToDarken)

class Button(Element):
    def Update(self, mouseJustClicked: bool):
        p = pygame.mouse.get_pos()

        cursorInsideBounds = p[0] in range(self.x, self.x + self.w) and p[1] in range(self.y, self.y+self.h)

        # Change colour based on if hovered over or not
        self.currentColour = self.defaultColour

        if cursorInsideBounds:
            self.currentColour = self._darken(self.currentColour, 0.8)

            # Darken even more if just clicked
            if mouseJustClicked:
                self.Clicked()

        self.Draw()

    def Clicked(self):
        return
    
class PieceButton(Button):
    def __init__(self, piece):
        self.piece = piece

        self.selected = False

        self.GetImage()

        super().__init__(piece.square[0] * 45, piece.square[1] * 45, 45, 45)

    def GetImage(self):
        # Find the right image for this piece
        pieceIndex = self.piece.pieceType.value
        if self.piece.isWhite: pieceIndex += 6

        self.image = pieceImages[pieceIndex]

    def Draw(self) -> None:
        win.blit(self.image, (self.x, self.y))

        if self.selected: self.DrawPossibleMoves()

    def DrawPossibleMoves(self) -> None:
        for move in self.piece.possibleMoves:
            p = move.targetSquare

            screenPos = [x*45 + 22 for x in p]

            gfxdraw.aacircle(win, screenPos[0], screenPos[1], 4, (60, 70, 80))
            gfxdraw.filled_circle(win, screenPos[0], screenPos[1], 4, (60, 70, 80))

            # Draw red for captures
            if self.piece.boardReference.GetPieceAtSquare(p) != None or move.moveType == MoveType.EN_PASSANT:
                transparentSurface = pygame.Surface((45, 45), pygame.SRCALPHA)
                transparentSurface.set_alpha(128)
                transparentSurface.fill((200,30,40))

                win.blit(transparentSurface, (screenPos[0] - 22, screenPos[1] - 22))

    def Clicked(self) -> None:
        # Make sure its the right turn
        if self.piece.boardReference.isWhiteTurn != self.piece.isWhite: return

        self.selected = not self.selected

        if self.selected:
            self.piece.CalculatePossibleMoves()
        else:
            self.piece.possibleMoves = None

    def Update(self, mouseJustClicked: bool):
        self.x, self.y = self.piece.square[0] * 45, self.piece.square[1] * 45

        if self.piece.isCaptured: 
            self.Draw()
            return

        if self.piece.justPromoted:
            self.GetImage()

            self.piece.justPromoted = False

        if mouseJustClicked:
            # See if this is somewhere that we can move 
            if self.selected:
                for move in self.piece.possibleMoves:
                    p = move.targetSquare

                    screenPos = [x*45 for x in p]
                    m = pygame.mouse.get_pos()

                    cursorInsideBounds = m[0] in range(screenPos[0], screenPos[0] + 45) and m[1] in range(screenPos[1], screenPos[1] + 45)

                    if cursorInsideBounds:
                        self.piece.boardReference.MakeMove(move)
                        break

            self.selected = False

        super().Update(mouseJustClicked)

class Piece():
    def __init__(self, pieceType: PieceType, isWhite: bool, square: Square, board: Board):
        self.square = square

        self.pieceType = pieceType
        self.isWhite = isWhite

        self.timesMoved = 0

        self.possibleMoves = None
        self.isCaptured = False

        self.justPromoted = False

        self.boardReference = board
        board.AddPiece(self, square)
    
    def CalculatePossibleMoves(self) -> None:
        self.possibleMoves = findLegalMovesForPiece(self.boardReference, self)

    def MakeMove(self, move, unmake: bool = False) ->  None:
        newSquare = move.targetSquare

        self.square = newSquare

        # Decrement if this is an unmake
        self.timesMoved += -1 if unmake else 1

    def Capture(self):
        self.isCaptured = True

        self.square = (0, 8)

    def Promote(self, newPieceType: PieceType):
        self.pieceType = newPieceType

        self.justPromoted = not self.justPromoted
    
# List containing all UI elements
elements = []
# endregion

isWhiteTurn = True

def drawGrid() -> None:
    count = 0

    for y in range(8):
        for x in range(8):
            # Change colour alternating
            count += 1
            cellColour = (181, 135, 98)

            if count % 2 == 1:
                cellColour = (239, 218, 180)

            cellPos = (x*45, y*45)
            pygame.draw.rect(win, cellColour, (cellPos[0], cellPos[1], 45, 45))

        count += 1

# Spawn in all the boards pieces from a FEN position
def spawnPieces(board: Board, FEN: str) -> None:
    ranks = FEN.split("/")

    for i in range(len(ranks)):
        rank = ranks[i]

        currentSquare = 0 # Track current position
        currentIndex = -1 # Track index of string

        while currentSquare < 8:
            currentIndex += 1

            letter = rank[currentIndex]
            isWhite = False

            # Skip these squares
            if letter.isnumeric():
                currentSquare += int(letter)
                continue
        
            if letter.isupper():
                isWhite = True

            # Add the piece
            pieceType = letterToPiece[letter.lower()]
            piece = Piece(pieceType, isWhite, (currentSquare, i), board)

            board.pieces.append(piece)

            # Add the ui
            pieceButton = PieceButton(piece)
            elements.append(pieceButton)

            currentSquare += 1
    
def triggerCheckmate(board: Board):
    board.checkmate = True

    checkmateBox = Button(30, 30, 300, 300, (36, 41, 46))
    elements.append(checkmateBox)
# endregion

# region Bot
class Move:
    def __init__(self, piece: Piece, startSquare: Square, targetSquare: Square, capturedPiece: Piece, moveType: MoveType):
        self.piece = piece
        self.startSquare = startSquare
        self.targetSquare = targetSquare
        self.capturedPiece = capturedPiece
        self.moveType = moveType

        self.putsOtherKingInCheck = False

    def ToNotation(self):
        pieceLetter = pieceToLetter[self.piece.pieceType]

        if pieceLetter == "p": pieceLetter = ""

        moveString = f"{len(board.moveHistory)}. "

        if self.piece.isWhite: 
            pieceLetter = pieceLetter.upper()
        else:
            moveString += "... "

        if self.moveType == MoveType.SHORT_CASTLE:
            moveString += "O-O"
        elif self.moveType == MoveType.LONG_CASTLE:
            moveString += "O-O-O"
        else:
            moveString += f"{pieceLetter}{SquareToNotation(self.targetSquare)}"

        if self.moveType == MoveType.QUEEN_PROMOTION: moveString += "=Q"
        if self.moveType == MoveType.ROOK_PROMOTION: moveString += "=R"
        if self.moveType == MoveType.KNIGHT_PROMOTION: moveString += "=N"
        if self.moveType == MoveType.BISHOP_PROMOTION: moveString += "=B"

        if self.putsOtherKingInCheck: moveString += "+"

        return moveString

#Move = tuple[Piece, Square, Square, Piece, MoveType]
def isValidMove(board: Board, move: Move, evaluateInCheck: bool = True) -> bool:
    # Make sure this isn't off the board
    piece = move.piece
    targetSquare = move.targetSquare
    capturePiece = move.capturedPiece
    moveType = move.moveType

    castling = False

    # Castling
    if moveType == MoveType.SHORT_CASTLE:
        # Check if there is a clear path
        if not len(checkAlongDirection(board, piece, (1, 0))) == 2: return False
        
        # Check if the rook is there/can move
        if capturePiece == None or capturePiece.timesMoved > 0: return False

        # Also make sure we can move just 1 over
        throughCheck = not isValidMove(board, Move(piece, move.startSquare, AddSquares(targetSquare, (-1, 0)), None, MoveType.NORMAL))

        if throughCheck: return False

        castling = True
    
    elif moveType == MoveType.LONG_CASTLE:
        if not len(checkAlongDirection(board, piece, (-1, 0))) == 3: return False

        if capturePiece == None or capturePiece.timesMoved > 0: return False

        throughCheck = not isValidMove(board, Move(piece, move.startSquare, AddSquares(targetSquare, (1, 0)), None, MoveType.NORMAL))

        if throughCheck: return False

        castling = True
    
    elif moveType == MoveType.EN_PASSANT:
        return True

    if not (0 <= targetSquare[0] < 8) or not (0 <= targetSquare[1] < 8):
        return False

    # Make sure the square isn't occupied by a piece of the same colour unless castling
    if capturePiece != None and piece.isWhite == capturePiece.isWhite and not castling:
        return False

    # Make sure this doesn't put us in check
    if capturePiece != None and capturePiece.pieceType == PieceType.KING:
        move.putsOtherKingInCheck = piece.isWhite != capturePiece.isWhite

    if evaluateInCheck:
        badMove = False

        board.MakeMove(move)
        opponentMoves = findLegalMoves(board, evaluateInCheck = False)
        
        for m in opponentMoves:
            #print(f"{m.ToNotation()} - currently {'white' if board.isWhiteTurn else 'black'}'s turn")

            if m.putsOtherKingInCheck: 
                badMove = True
                break

        board.UnmakeLastMove()

        if badMove: return False

    return True

def checkAlongDirection(board: Board, piece: Piece, moveDir: Move) -> list[Square]: # list of available squares
    collidedPiece = None
    currentSquare = piece.square
    availableSquares = []

    for i in range(8):
        currentSquare = AddSquares(currentSquare, moveDir)
        collidedPiece = board.GetPieceAtSquare(currentSquare)

        if collidedPiece == None:
            availableSquares.append(currentSquare)
        else:
            if collidedPiece.isWhite != piece.isWhite:
                availableSquares.append(currentSquare)
            break
    
    return availableSquares

def findLegalMoves(board: Board, evaluateInCheck: bool = True) -> list[Move]:
    # loop over all pieces
    # find where they can move
    legalMovesInPosition = []

    for piece in board.pieces:
        # Only get the right coloured pieces
        if piece.isWhite != board.isWhiteTurn: continue

        legalMovesInPosition.extend(findLegalMovesForPiece(board, piece, evaluateInCheck))

    if len(legalMovesInPosition) == 0:
        triggerCheckmate(board)

    return legalMovesInPosition

def findLegalMovesForPiece(board: Board, piece: Piece, evaluateInCheck: bool = True) -> list[Move]:
    # get all moves for this piece
    # include en passant and castling
    # make sure that square isn't blocked
    # make sure it doesn't lead to being in check

    if piece.isCaptured: return []

    # Get all moves for this piece
    possibleSquaresToMoveTo = []
    validMoves = []
    pos = piece.square

    #  Handle each piece type manually
    if piece.pieceType == PieceType.KNIGHT:
        KNIGHT_MOVES = [(-1, -2), (1, -2), (-2, -1), (2, -1), (-2, 1), (2, 1), (-1, 2), (1, 2)]

        possibleSquaresToMoveTo.extend([AddSquares(pos, m) for m in KNIGHT_MOVES])

    elif piece.pieceType == PieceType.KING:
        KING_MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        possibleSquaresToMoveTo.extend([AddSquares(pos, m) for m in KING_MOVES])

        # Castling
        if piece.timesMoved == 0:
            shortCastleMove = Move(piece, pos, AddSquares(pos, (2, 0)), board.GetPieceAtSquare(AddSquares(pos, (3, 0))), MoveType.SHORT_CASTLE)
            longCastleMove = Move(piece, pos, AddSquares(pos, (-2, 0)), board.GetPieceAtSquare(AddSquares(pos, (-4, 0))), MoveType.LONG_CASTLE)

            if isValidMove(board, shortCastleMove, evaluateInCheck): validMoves.append(shortCastleMove)
            if isValidMove(board, longCastleMove, evaluateInCheck): validMoves.append(longCastleMove)

    elif piece.pieceType == PieceType.PAWN:
        direction = -1 if piece.isWhite else 1

        oneAhead = AddSquares(pos, (0, direction))
        oneAheadPiece = board.GetPieceAtSquare(oneAhead)

        atSecondLastRank = (piece.square[1] == 1 and piece.isWhite) or (piece.square[1] == 6 and not piece.isWhite)

        if oneAheadPiece == None:
            # Handle promotions
            if atSecondLastRank:
                promoteMoves = [Move(piece, pos, oneAhead, None, p) for p in promotionMoveTypes]

                validMoves.extend([m for m in promoteMoves if isValidMove(board, m, evaluateInCheck)])
            else:
                possibleSquaresToMoveTo.append(oneAhead)

            # Two on first move
            if piece.timesMoved == 0:
                twoAhead = AddSquares(pos, (0, direction * 2))
                twoAheadPiece = board.GetPieceAtSquare(twoAhead)

                if twoAheadPiece == None: 
                    possibleSquaresToMoveTo.append(twoAhead)

        # Diagonal captures
        for i in [-1, 1]:
            diagonal = AddSquares(pos, (i, direction))
            capturePiece = board.GetPieceAtSquare(diagonal)

            if capturePiece != None:
                # Promotions
                if atSecondLastRank:
                    promoteMoves = [Move(piece, pos, diagonal, capturePiece, p) for p in promotionMoveTypes]

                    validMoves.extend([m for m in promoteMoves if isValidMove(board, m, evaluateInCheck)])
                else:
                    possibleSquaresToMoveTo.append(diagonal)

        # En passant
        if len(board.moveHistory) > 0:
            lastMove: Move = board.moveHistory[-1]

            # Check if left or right was last moved
            if lastMove.piece in [board.GetPieceAtSquare(AddSquares(piece.square, (-1, 0))), board.GetPieceAtSquare(AddSquares(piece.square, (1, 0)))]:
                # Check if its a double moved pawn
                if lastMove.piece.pieceType == PieceType.PAWN and lastMove.piece.timesMoved == 1:
                    captureSquare = AddSquares(lastMove.piece.square, (0, -1))

                    if not piece.isWhite:
                        captureSquare = AddSquares(lastMove.piece.square, (0, 1))

                    enPassantMove = Move(piece, piece.square, captureSquare, lastMove.piece, MoveType.EN_PASSANT)

                    if isValidMove(board, enPassantMove, evaluateInCheck): validMoves.append(enPassantMove)

    elif piece.pieceType == PieceType.ROOK:
        for direction in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
            possibleSquaresToMoveTo.extend(checkAlongDirection(board, piece, direction))

    elif piece.pieceType == PieceType.BISHOP:
        for direction in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            possibleSquaresToMoveTo.extend(checkAlongDirection(board, piece, direction))

    elif piece.pieceType == PieceType.QUEEN:
        for direction in [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, -1), (0, 1)]:
            possibleSquaresToMoveTo.extend(checkAlongDirection(board, piece, direction))
    
    # Add only the valid ones
    for square in possibleSquaresToMoveTo:
        move = Move(piece, piece.square, square, board.GetPieceAtSquare(square), MoveType.NORMAL)

        if isValidMove(board, move, evaluateInCheck):
            validMoves.append(move)
    
    return validMoves

# Return an eval as a number
def evaluatePosition(board: Board) -> int:
    # get material
    whiteMaterial = 0
    blackMaterial = 0

    # Just add up all material values
    for piece in board.pieces:
        if piece.isCaptured: continue

        if piece.isWhite: whiteMaterial += pieceValues[piece.pieceType]
        else: blackMaterial += pieceValues[piece.pieceType]

    return whiteMaterial - blackMaterial

def makeComputerMove(board: Board) -> None:
    # generate possible moves
    # evaluate
    # pick the best
    # make the move
    moves = findLegalMoves(board)

    bestMove: Move = None
    bestEval = -999999

    for move in moves:
        board.MakeMove(move)
        evaluation = -evaluatePosition(board)
        board.UnmakeLastMove()

        if board.isWhiteTurn: evaluation *= -1

        if evaluation >= bestEval:
            bestMove = move
            bestEval = evaluation

    if len(moves) > 0:
        board.MakeMove(bestMove)

# endregion

# Initialise everything
def init() -> Board:
    board = Board()

    spawnPieces(board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR") #rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR

    return board

# Handle pygame events like mouse down and return the mouse state
def handleEvents() -> bool:
    mouseJustClicked = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouseJustClicked = True

    return mouseJustClicked

def updateScreen(mouseJustClicked) -> None:
    # Update all UI
    for e in elements:
        e.Update(mouseJustClicked)

    pygame.display.update()
    clock.tick(60)

# Main game loop
def game(board: Board) -> None:
    if board.checkmate:
        return

    # Assume white is player
    if board.isWhiteTurn:
        pass

    # Do computer stuff
    else:
        makeComputerMove(board)

# Start everything up
board = init()

# Game loop
while True:
    win.fill((240, 250, 250))
    drawGrid()
    
    # Do Game
    game(board)

    mouseJustClicked = handleEvents()
    updateScreen(mouseJustClicked)