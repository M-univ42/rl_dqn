import matplotlib.pyplot as plt
import json

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()

def plot_dqn_training(input_path,output_path):
    print("start")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    with open(input_path, "r") as f:
        data = json.load(f)
    

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(data["rewards"], color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('Epsilon', color=color)  
    ax2.plot(data["epsilons"], color=color, label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    print("mid")
    plt.title('DQN Training Progress')
    plt.grid()
    plt.savefig(output_path)
    plt.show()
    print("end")
plot_dqn_training("training_log.json","training_plot.png")