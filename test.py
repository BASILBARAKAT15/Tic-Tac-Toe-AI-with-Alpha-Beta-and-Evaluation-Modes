import customtkinter as ctk
import tkinter as tk
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tkinter import messagebox

# --------------------------
# Board Class
# --------------------------
class Board:
    def __init__(self):
        self.board = [' ']*9

    def available_moves(self):
        return [i for i, c in enumerate(self.board) if c==' ']

    def make_move(self, idx, player):
        if self.board[idx]==' ':
            self.board[idx]=player
            return True
        return False

    def undo_move(self, idx):
        self.board[idx]=' '

    def get_winner(self):
        lines=[[0,1,2],[3,4,5],[6,7,8],
               [0,3,6],[1,4,7],[2,5,8],
               [0,4,8],[2,4,6]]
        for line in lines:
            if self.board[line[0]]==self.board[line[1]]==self.board[line[2]]!=' ':
                return self.board[line[0]], line
        return None, None

    def is_terminal(self):
        winner, _ = self.get_winner()
        if winner: return True, winner
        elif ' ' not in self.board: return True, 'Draw'
        return False, None

# --------------------------
# Heuristic Evaluation
# --------------------------
def heuristic(board:Board, player='X'):
    score=0
    lines=[[0,1,2],[3,4,5],[6,7,8],
           [0,3,6],[1,4,7],[2,5,8],
           [0,4,8],[2,4,6]]
    for line in lines:
        x=sum(1 for i in line if board.board[i]=='X')
        o=sum(1 for i in line if board.board[i]=='O')
        e=sum(1 for i in line if board.board[i]==' ')
        if x==3: score+=100
        elif o==3: score-=100
        elif x==2 and e==1: score+=10
        elif o==2 and e==1: score-=10
    return score if player=='X' else -score

# --------------------------
# ML Feature Extraction
# --------------------------
def extract_features(board:Board):
    b=board.board
    num_X=b.count('X'); num_O=b.count('O')
    lines=[[0,1,2],[3,4,5],[6,7,8],
           [0,3,6],[1,4,7],[2,5,8],
           [0,4,8],[2,4,6]]
    X_two=sum(1 for l in lines if sum(1 for i in l if b[i]=='X')==2 and sum(1 for i in l if b[i]==' ')==1)
    O_two=sum(1 for l in lines if sum(1 for i in l if b[i]=='O')==2 and sum(1 for i in l if b[i]==' ')==1)
    X_center=1 if b[4]=='X' else 0; O_center=1 if b[4]=='O' else 0
    corners=[0,2,6,8]; X_corners=sum(1 for c in corners if b[c]=='X'); O_corners=sum(1 for c in corners if b[c]=='O')
    return np.array([num_X,num_O,X_two,O_two,X_center,O_center,X_corners,O_corners]).reshape(1,-1)

# --------------------------
# Load ML Model
# --------------------------
def train_ml_model(path=r"C:\Users\HP\Downloads\tictactoe_dataset.csv"):
    data=pd.read_csv(path)
    X=data.drop('label',axis=1); y=data['label']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)
    print("ML model trained. R^2:", model.score(X_test,y_test))
    return model

ml_model=train_ml_model()

def ml_eval(board:Board,player='X'):
    features=extract_features(board)
    score=ml_model.predict(features)[0]
    return score if player=='X' else -score

# --------------------------
# Alpha-Beta Pruning
# --------------------------
def alpha_beta(board,depth,alpha,beta,maximizing,eval_fn,player='X'):
    terminal,winner=board.is_terminal()
    if depth==0 or terminal:
        if terminal:
            if winner==player: return 1000
            elif winner=='Draw': return 0
            else: return -1000
        return eval_fn(board,player)
    if maximizing:
        max_eval=-float('inf')
        for move in board.available_moves():
            board.make_move(move,player)
            eval=alpha_beta(board,depth-1,alpha,beta,False,eval_fn,player)
            board.undo_move(move)
            max_eval=max(max_eval,eval)
            alpha=max(alpha,eval)
            if beta<=alpha: break
        return max_eval
    else:
        min_eval=float('inf')
        opponent='O' if player=='X' else 'X'
        for move in board.available_moves():
            board.make_move(move,opponent)
            eval=alpha_beta(board,depth-1,alpha,beta,True,eval_fn,player)
            board.undo_move(move)
            min_eval=min(min_eval,eval)
            beta=min(beta,eval)
            if beta<=alpha: break
        return min_eval

# --------------------------
# AI Move
# --------------------------
def get_ai_move(board,eval_fn,player='X',difficulty='Hard'):
    moves=board.available_moves()
    if difficulty=='Easy': return random.choice(moves),{}
    depth=2 if difficulty=='Normal' else 9
    best_score=-float('inf'); best_move=None; scores={}
    for move in moves:
        board.make_move(move,player)
        score=alpha_beta(board,depth-1,-float('inf'),float('inf'),False,eval_fn,player)
        board.undo_move(move)
        scores[move]=score
        if score>best_score: best_score=score; best_move=move
    return best_move,scores

# --------------------------
# GUI Application
# --------------------------
class TicTacToeGUI:
    COLORS={'X':'#1f77b4','O':'#d62728','highlight':'#2ca02c',
            'board':'#f7f7f7','panel':'#e0f0ff'}

    def __init__(self,root):
        self.root=root
        self.root.title("Tic-Tac-Toe Pro")
        self.root.geometry("600x650")
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.main_frame=ctk.CTkFrame(self.root, fg_color="#ffffff")
        self.main_frame.grid(row=0,column=0,sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=0)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.board=Board()
        self.buttons=[]
        self.player_choice='X'; self.ai_player='O'
        self.difficulty='Hard'; self.eval_fn=heuristic

        self.create_option_panel()
        self.create_board()

    # --------------------------
    # Option Panel
    # --------------------------
    def create_option_panel(self):
        self.option_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color=self.COLORS['panel'])
        self.option_frame.grid(row=0,column=0,sticky="nsew", padx=20, pady=10)
        self.option_frame.grid_columnconfigure((0,1,2),weight=1)

        ctk.CTkLabel(self.option_frame,text="Tic-Tac-Toe Settings",
                     font=("Segoe UI",16,"bold")).grid(row=0,column=0,columnspan=3,pady=5)

        # Side Selection
        self.side_var = tk.StringVar(value='X')
        ctk.CTkLabel(self.option_frame,text="Side:", font=("Segoe UI",12,"bold")).grid(row=1,column=0)
        self.side_buttons=[]
        for val in ['X','O']:
            btn = ctk.CTkButton(self.option_frame,text=val,
                                command=lambda v=val:self.set_side(v))
            btn.grid(row=1,column=1 if val=='X' else 2,padx=5,pady=5)
            self.side_buttons.append(btn)
        self.update_side_buttons()

        # Difficulty Selection
        self.diff_var = tk.StringVar(value='Hard')
        ctk.CTkLabel(self.option_frame,text="Difficulty:", font=("Segoe UI",12,"bold")).grid(row=2,column=0)
        self.diff_buttons=[]
        for i,d in enumerate(['Easy','Normal','Hard']):
            btn = ctk.CTkButton(self.option_frame,text=d,command=lambda x=d:self.set_difficulty(x))
            btn.grid(row=2,column=i,padx=5,pady=5)
            self.diff_buttons.append(btn)
        self.update_diff_buttons()

        # Evaluation Selection
        self.eval_var = tk.StringVar(value='Heuristic')
        ctk.CTkLabel(self.option_frame,text="Evaluation:", font=("Segoe UI",12,"bold")).grid(row=3,column=0)
        self.eval_buttons=[]
        for i,e in enumerate(['Heuristic','ML']):
            btn = ctk.CTkButton(self.option_frame,text=e,command=lambda x=e:self.set_eval(x))
            btn.grid(row=3,column=i,padx=5,pady=5)
            self.eval_buttons.append(btn)
        self.update_eval_buttons()

        # Start Button
        ctk.CTkButton(self.option_frame,text="Start Game",command=self.start_game).grid(row=4,column=0,columnspan=3,pady=5)

    # --------------------------
    # Option Updates
    # --------------------------
    def set_side(self,val):
        self.side_var.set(val)
        self.update_side_buttons()

    def update_side_buttons(self):
        for btn in self.side_buttons:
            if btn.cget('text')==self.side_var.get():
                btn.configure(fg_color="#2ca02c", text_color="white")
            else:
                btn.configure(fg_color="#f0f0f0", text_color="black")

    def set_difficulty(self,diff):
        self.difficulty=diff
        self.update_diff_buttons()

    def update_diff_buttons(self):
        for btn in self.diff_buttons:
            if btn.cget('text')==self.difficulty:
                btn.configure(fg_color="#2ca02c", text_color="white")
            else:
                btn.configure(fg_color="#f0f0f0", text_color="black")

    def set_eval(self,e):
        self.eval_fn=ml_eval if e=='ML' else heuristic
        self.eval_var.set(e)
        self.update_eval_buttons()

    def update_eval_buttons(self):
        for btn in self.eval_buttons:
            if btn.cget('text')==self.eval_var.get():
                btn.configure(fg_color="#2ca02c", text_color="white")
            else:
                btn.configure(fg_color="#f0f0f0", text_color="black")

    # --------------------------
    # Board GUI
    # --------------------------
    def create_board(self):
        self.board_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color=self.COLORS['board'])
        self.board_frame.grid(row=1,column=0,sticky="nsew", padx=20,pady=20)
        for i in range(3):
            self.board_frame.grid_rowconfigure(i,weight=1)
            self.board_frame.grid_columnconfigure(i,weight=1)

        for i in range(9):
            btn=ctk.CTkButton(self.board_frame,text='',font=("Arial Black",28,"bold"),
                               command=lambda i=i:self.player_move(i))
            btn.grid(row=i//3,column=i%3,sticky="nsew", padx=5,pady=5)
            self.buttons.append(btn)

    # --------------------------
    # Start Game
    # --------------------------
    def start_game(self):
        self.board=Board()
        for btn in self.buttons: btn.configure(text='',fg_color="#f0f0f0", state='normal')
        self.player_choice=self.side_var.get()
        self.ai_player='O' if self.player_choice=='X' else 'X'
        if self.ai_player=='X': self.root.after(500,self.ai_move)

    # --------------------------
    # Player move
    # --------------------------
    def player_move(self,idx):
        if self.board.make_move(idx,self.player_choice):
            self.update_buttons()
            term,winner=self.board.is_terminal()
            if term: self.end_game(winner); return
            self.root.after(300,self.ai_move)

    # --------------------------
    # AI move
    # --------------------------
    def ai_move(self):
        move,scores=get_ai_move(self.board,self.eval_fn,self.ai_player,self.difficulty)
        self.board.make_move(move,self.ai_player)
        self.update_buttons()
        term,winner=self.board.is_terminal()
        if term: self.end_game(winner)

    # --------------------------
    # Update GUI
    # --------------------------
    def update_buttons(self):
        for i,btn in enumerate(self.buttons):
            btn.configure(text=self.board.board[i])
            if self.board.board[i]=='X': btn.configure(fg_color="#1f77b4", text_color="white")
            elif self.board.board[i]=='O': btn.configure(fg_color="#d62728", text_color="white")
            else: btn.configure(fg_color="#f0f0f0", text_color="black")

    # --------------------------
    # End Game
    # --------------------------
    def end_game(self,winner):
        if winner=='Draw': messagebox.showinfo("Game Over","It's a draw!")
        else:
            messagebox.showinfo("Game Over",f"{winner} wins!")
            _,line=self.board.get_winner()
            if line:
                for idx in line: self.buttons[idx].configure(fg_color="#2ca02c")
        for btn in self.buttons: btn.configure(state='disabled')


# --------------------------
# Run Application
# --------------------------
if __name__=="__main__":
    root=ctk.CTk()
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    app=TicTacToeGUI(root)
    root.mainloop()
