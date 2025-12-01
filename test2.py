import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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
            'bg':'#e0f2f7', 'board':'#d0e7f9', 'btn_bg':'#f0f0f0', 'panel':'#ffffff'}
    
    def __init__(self,root):
        self.root=root
        self.root.title("Tic-Tac-Toe AI Pro")
        self.root.configure(bg=self.COLORS['bg'])
        self.board=Board()
        self.buttons=[]; self.score_labels=[]
        self.player_choice='X'; self.ai_player='O'
        self.difficulty='Hard'; self.eval_fn=heuristic
        self.create_option_panel()
        self.create_board()

    # --------------------------
    # Professional Option Panel
    # --------------------------
    def create_option_panel(self):
        self.option_frame = tk.Frame(self.root, bg=self.COLORS['panel'], bd=3, relief='raised', padx=10, pady=10)
        self.option_frame.pack(pady=15, padx=15, fill='x')

        tk.Label(self.option_frame, text="Game Settings", bg=self.COLORS['panel'],
                 font=('Helvetica', 14, 'bold')).grid(row=0, column=0, columnspan=8, pady=(5,10))

        # Side selection
        tk.Label(self.option_frame, text="Choose Side:", bg=self.COLORS['panel'], font=('Arial',12,'bold')).grid(row=1,column=0,padx=10,pady=5)
        self.side_var = tk.StringVar(value='X')
        tk.Radiobutton(self.option_frame,text='X',variable=self.side_var,value='X',bg=self.COLORS['panel'],font=('Arial',12)).grid(row=1,column=1)
        tk.Radiobutton(self.option_frame,text='O',variable=self.side_var,value='O',bg=self.COLORS['panel'],font=('Arial',12)).grid(row=1,column=2)

        # Difficulty selection
        tk.Label(self.option_frame, text="Difficulty:", bg=self.COLORS['panel'], font=('Arial',12,'bold')).grid(row=1,column=3,padx=10)
        self.diff_buttons=[]
        for i,diff in enumerate(['Easy','Normal','Hard']):
            b=tk.Button(self.option_frame,text=diff,width=8,bg=self.COLORS['btn_bg'],font=('Arial',11),
                        relief='raised', command=lambda d=diff:self.set_difficulty(d))
            b.grid(row=1,column=4+i,padx=5)
            self.diff_buttons.append(b)
        self.set_difficulty('Hard')

        # Evaluation selection
        tk.Label(self.option_frame, text="Evaluation:", bg=self.COLORS['panel'], font=('Arial',12,'bold')).grid(row=2,column=0,padx=10,pady=5)
        self.eval_var=tk.StringVar(value='Heuristic')
        self.eval_buttons=[]
        for i,eval_type in enumerate(['Heuristic','ML']):
            b=tk.Button(self.option_frame,text=eval_type,width=10,bg=self.COLORS['btn_bg'],font=('Arial',11),
                        relief='raised', command=lambda e=eval_type:self.set_eval(e))
            b.grid(row=2,column=1+i,padx=5, pady=5)
            self.eval_buttons.append(b)
        self.set_eval('Heuristic')

        # Start Game
        tk.Button(self.option_frame,text="Start Game",bg='#ff7f0e',fg='white',font=('Arial',12,'bold'),
                  relief='raised', command=self.start_game).grid(row=2,column=4, padx=10)

    def set_difficulty(self,diff):
        self.difficulty=diff
        for btn in self.diff_buttons:
            btn.config(bg='#4caf50' if btn['text']==diff else self.COLORS['btn_bg'])

    def set_eval(self,eval_type):
        self.eval_fn = ml_eval if eval_type=='ML' else heuristic
        self.eval_var.set(eval_type)
        for btn in self.eval_buttons:
            btn.config(bg='#4caf50' if btn['text']==eval_type else self.COLORS['btn_bg'])

    # --------------------------
    # Board GUI
    # --------------------------
    def create_board(self):
        board_frame = tk.Frame(self.root, bg=self.COLORS['board'], bd=5, relief='ridge', padx=10, pady=10)
        board_frame.pack(pady=10, padx=15)

        for i in range(9):
            btn=tk.Button(board_frame,text=' ',width=6,height=3,font=('Arial',24),
                          command=lambda i=i:self.player_move(i),
                          bg=self.COLORS['btn_bg'],activebackground='#e0e0e0',relief='raised', bd=3)
            btn.grid(row=i//3,column=i%3,padx=5,pady=5)
            self.buttons.append(btn)
            lbl=tk.Label(board_frame,text='',font=('Arial',10),bg=self.COLORS['board'])
            lbl.grid(row=i//3,column=i%3,sticky='s')
            self.score_labels.append(lbl)

    # --------------------------
    # Start Game
    # --------------------------
    def start_game(self):
        self.board=Board()
        for btn in self.buttons: btn.config(text=' ',state='normal',bg=self.COLORS['btn_bg'])
        for lbl in self.score_labels: lbl.config(text='')
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
        self.update_buttons(scores)
        term,winner=self.board.is_terminal()
        if term: self.end_game(winner)

    # --------------------------
    # Update GUI
    # --------------------------
    def update_buttons(self,scores=None):
        for i,btn in enumerate(self.buttons):
            btn.config(text=self.board.board[i])
            if self.board.board[i]=='X': btn.config(fg=self.COLORS['X'])
            elif self.board.board[i]=='O': btn.config(fg=self.COLORS['O'])
            else: btn.config(fg='black')
            if scores and self.board.board[i]==' ':
                self.score_labels[i].config(text=f"{scores.get(i,0):.1f}")
            else:
                self.score_labels[i].config(text='')

    # --------------------------
    # End Game
    # --------------------------
    def end_game(self,winner):
        if winner=='Draw': messagebox.showinfo("Game Over","It's a draw!")
        else:
            messagebox.showinfo("Game Over",f"{winner} wins!")
            _,line=self.board.get_winner()
            if line:
                for idx in line: self.buttons[idx].config(bg=self.COLORS['highlight'])
        for btn in self.buttons: btn.config(state='disabled')

# --------------------------
# Run Application
# --------------------------
if __name__=="__main__":
    root=tk.Tk()
    app=TicTacToeGUI(root)
    root.mainloop()
    
    
    
    
    
    
    
    
    
    
#     import tkinter as tk
# from tkinter import messagebox
# import numpy as np
# import pandas as pd
# import random
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

# # --------------------------
# # Board Class (same as before)
# # --------------------------
# class Board:
#     def __init__(self):
#         self.board = [' ']*9
#     def available_moves(self):
#         return [i for i, c in enumerate(self.board) if c==' ']
#     def make_move(self, idx, player):
#         if self.board[idx]==' ':
#             self.board[idx]=player
#             return True
#         return False
#     def undo_move(self, idx):
#         self.board[idx]=' '
#     def get_winner(self):
#         lines=[[0,1,2],[3,4,5],[6,7,8],
#                [0,3,6],[1,4,7],[2,5,8],
#                [0,4,8],[2,4,6]]
#         for line in lines:
#             if self.board[line[0]]==self.board[line[1]]==self.board[line[2]]!=' ':
#                 return self.board[line[0]], line
#         return None, None
#     def is_terminal(self):
#         winner, _ = self.get_winner()
#         if winner: return True, winner
#         elif ' ' not in self.board: return True, 'Draw'
#         return False, None

# # --------------------------
# # Heuristic Evaluation
# # --------------------------
# def heuristic(board:Board, player='X'):
#     score=0
#     lines=[[0,1,2],[3,4,5],[6,7,8],
#            [0,3,6],[1,4,7],[2,5,8],
#            [0,4,8],[2,4,6]]
#     for line in lines:
#         x=sum(1 for i in line if board.board[i]=='X')
#         o=sum(1 for i in line if board.board[i]=='O')
#         e=sum(1 for i in line if board.board[i]==' ')
#         if x==3: score+=100
#         elif o==3: score-=100
#         elif x==2 and e==1: score+=10
#         elif o==2 and e==1: score-=10
#     return score if player=='X' else -score

# # --------------------------
# # ML Feature Extraction
# # --------------------------
# def extract_features(board:Board):
#     b=board.board
#     num_X=b.count('X'); num_O=b.count('O')
#     lines=[[0,1,2],[3,4,5],[6,7,8],
#            [0,3,6],[1,4,7],[2,5,8],
#            [0,4,8],[2,4,6]]
#     X_two=sum(1 for l in lines if sum(1 for i in l if b[i]=='X')==2 and sum(1 for i in l if b[i]==' ')==1)
#     O_two=sum(1 for l in lines if sum(1 for i in l if b[i]=='O')==2 and sum(1 for i in l if b[i]==' ')==1)
#     X_center=1 if b[4]=='X' else 0; O_center=1 if b[4]=='O' else 0
#     corners=[0,2,6,8]; X_corners=sum(1 for c in corners if b[c]=='X'); O_corners=sum(1 for c in corners if b[c]=='O')
#     return np.array([num_X,num_O,X_two,O_two,X_center,O_center,X_corners,O_corners]).reshape(1,-1)

# # --------------------------
# # Load ML Model
# # --------------------------
# def train_ml_model(path=r"C:\Users\HP\Downloads\tictactoe_dataset.csv"):
#     data=pd.read_csv(path)
#     X=data.drop('label',axis=1); y=data['label']
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#     model=RandomForestRegressor(n_estimators=100,random_state=42)
#     model.fit(X_train,y_train)
#     print("ML model trained. R^2:", model.score(X_test,y_test))
#     return model

# ml_model=train_ml_model()

# def ml_eval(board:Board,player='X'):
#     features=extract_features(board)
#     score=ml_model.predict(features)[0]
#     return score if player=='X' else -score

# # --------------------------
# # Alpha-Beta Pruning
# # --------------------------
# def alpha_beta(board,depth,alpha,beta,maximizing,eval_fn,player='X'):
#     terminal,winner=board.is_terminal()
#     if depth==0 or terminal:
#         if terminal:
#             if winner==player: return 1000
#             elif winner=='Draw': return 0
#             else: return -1000
#         return eval_fn(board,player)
#     if maximizing:
#         max_eval=-float('inf')
#         for move in board.available_moves():
#             board.make_move(move,player)
#             eval=alpha_beta(board,depth-1,alpha,beta,False,eval_fn,player)
#             board.undo_move(move)
#             max_eval=max(max_eval,eval)
#             alpha=max(alpha,eval)
#             if beta<=alpha: break
#         return max_eval
#     else:
#         min_eval=float('inf')
#         opponent='O' if player=='X' else 'X'
#         for move in board.available_moves():
#             board.make_move(move,opponent)
#             eval=alpha_beta(board,depth-1,alpha,beta,True,eval_fn,player)
#             board.undo_move(move)
#             min_eval=min(min_eval,eval)
#             beta=min(beta,eval)
#             if beta<=alpha: break
#         return min_eval

# # --------------------------
# # AI Move
# # --------------------------
# def get_ai_move(board,eval_fn,player='X',difficulty='Hard'):
#     moves=board.available_moves()
#     if difficulty=='Easy': return random.choice(moves),{}
#     depth=2 if difficulty=='Normal' else 9
#     best_score=-float('inf'); best_move=None; scores={}
#     for move in moves:
#         board.make_move(move,player)
#         score=alpha_beta(board,depth-1,-float('inf'),float('inf'),False,eval_fn,player)
#         board.undo_move(move)
#         scores[move]=score
#         if score>best_score: best_score=score; best_move=move
#     return best_move,scores

# # --------------------------
# # Responsive GUI
# # --------------------------
# class TicTacToeGUI:
#     COLORS={'X':'#1f77b4','O':'#d62728','highlight':'#2ca02c',
#             'bg':'#e0f2f7', 'board':'#d0e7f9', 'btn_bg':'#f0f0f0', 'panel':'#ffffff'}

#     def __init__(self,root):
#         self.root=root
#         self.root.title("Tic-Tac-Toe AI Pro")
#         self.root.configure(bg=self.COLORS['bg'])
#         self.root.rowconfigure(1,weight=1)
#         self.root.columnconfigure(0,weight=1)

#         self.board=Board()
#         self.buttons=[]

#         self.player_choice='X'; self.ai_player='O'
#         self.difficulty='Hard'; self.eval_fn=heuristic

#         self.create_option_panel()
#         self.create_board()

#     def create_option_panel(self):
#         self.option_frame = tk.Frame(self.root, bg=self.COLORS['panel'], bd=3, relief='raised')
#         self.option_frame.grid(row=0,column=0,sticky='ew', padx=10, pady=10)
#         self.option_frame.columnconfigure(tuple(range(8)),weight=1)

#         tk.Label(self.option_frame,text="Game Settings", bg=self.COLORS['panel'], font=('Helvetica',14,'bold')).grid(row=0,column=0,columnspan=8)

#         # Side
#         self.side_var=tk.StringVar(value='X')
#         tk.Label(self.option_frame,text="Side:",bg=self.COLORS['panel']).grid(row=1,column=0)
#         tk.Radiobutton(self.option_frame,text='X',variable=self.side_var,value='X',bg=self.COLORS['panel'],command=self.start_game).grid(row=1,column=1)
#         tk.Radiobutton(self.option_frame,text='O',variable=self.side_var,value='O',bg=self.COLORS['panel'],command=self.start_game).grid(row=1,column=2)

#         # Difficulty
#         self.diff_buttons=[]
#         for i,diff in enumerate(['Easy','Normal','Hard']):
#             b=tk.Button(self.option_frame,text=diff,command=lambda d=diff:self.set_difficulty(d))
#             b.grid(row=1,column=3+i,sticky='ew', padx=2)
#             self.diff_buttons.append(b)
#         self.set_difficulty('Hard')

#         # Evaluation
#         self.eval_var=tk.StringVar(value='Heuristic')
#         self.eval_buttons=[]
#         for i,e in enumerate(['Heuristic','ML']):
#             b=tk.Button(self.option_frame,text=e,command=lambda ev=e:self.set_eval(ev))
#             b.grid(row=2,column=i,sticky='ew', padx=2, pady=5)
#             self.eval_buttons.append(b)
#         self.set_eval('Heuristic')

#         # Start Button
#         tk.Button(self.option_frame,text="Start Game",bg='#ff7f0e',fg='white',command=self.start_game).grid(row=2,column=3,columnspan=2,sticky='ew', padx=5)

#     def set_difficulty(self,diff):
#         self.difficulty=diff
#         for btn in self.diff_buttons:
#             btn.config(bg='#4caf50' if btn['text']==diff else self.COLORS['btn_bg'])

#     def set_eval(self,eval_type):
#         self.eval_fn=ml_eval if eval_type=='ML' else heuristic
#         self.eval_var.set(eval_type)
#         for btn in self.eval_buttons:
#             btn.config(bg='#4caf50' if btn['text']==eval_type else self.COLORS['btn_bg'])

#     def create_board(self):
#         self.board_frame = tk.Frame(self.root, bg=self.COLORS['board'], bd=3, relief='ridge')
#         self.board_frame.grid(row=1,column=0,sticky='nsew', padx=10, pady=10)
#         for i in range(3): self.board_frame.rowconfigure(i,weight=1)
#         for i in range(3): self.board_frame.columnconfigure(i,weight=1)

#         for i in range(9):
#             btn=tk.Button(self.board_frame,text=' ',font=('Arial',24),command=lambda i=i:self.player_move(i))
#             btn.grid(row=i//3,column=i%3,sticky='nsew', padx=5, pady=5)
#             self.buttons.append(btn)

#     def start_game(self):
#         self.board=Board()
#         for btn in self.buttons: btn.config(text=' ',bg=self.COLORS['btn_bg'],state='normal')
#         self.player_choice=self.side_var.get()
#         self.ai_player='O' if self.player_choice=='X' else 'X'
#         if self.ai_player=='X': self.root.after(500,self.ai_move)

#     def player_move(self,idx):
#         if self.board.make_move(idx,self.player_choice):
#             self.update_buttons()
#             term,winner=self.board.is_terminal()
#             if term: self.end_game(winner); return
#             self.root.after(300,self.ai_move)

#     def ai_move(self):
#         move,_=get_ai_move(self.board,self.eval_fn,self.ai_player,self.difficulty)
#         self.board.make_move(move,self.ai_player)
#         self.update_buttons()
#         term,winner=self.board.is_terminal()
#         if term: self.end_game(winner)

#     def update_buttons(self):
#         for i,btn in enumerate(self.buttons):
#             btn.config(text=self.board.board[i])
#             if self.board.board[i]=='X': btn.config(fg=self.COLORS['X'])
#             elif self.board.board[i]=='O': btn.config(fg=self.COLORS['O'])
#             else: btn.config(fg='black')

#     def end_game(self,winner):
#         if winner=='Draw': messagebox.showinfo("Game Over","It's a draw!")
#         else:
#             messagebox.showinfo("Game Over",f"{winner} wins!")
#             _,line=self.board.get_winner()
#             if line:
#                 for idx in line: self.buttons[idx].config(bg=self.COLORS['highlight'])
#         for btn in self.buttons: btn.config(state='disabled')


# if __name__=="__main__":
#     root=tk.Tk()
#     root.geometry("600x650")
#     root.rowconfigure(1,weight=1)
#     root.columnconfigure(0,weight=1)
#     app=TicTacToeGUI(root)
#     root.mainloop()

