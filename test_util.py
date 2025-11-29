import sys
from unittest.mock import MagicMock

# Mock torch since it's not in the environment
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()

import os
import shutil
# Now we can import util
from util import plot_training_metrics

def test_plot():
    print("Testing util.py plotting...")
    
    checkpoint_dir = './test_checkpoints'
    
    # Test Plotting
    print("Testing plot_training_metrics...")
    train_losses = [0.5, 0.4, 0.3]
    val_losses = [0.6, 0.5, 0.4]
    train_acc = [0.1, 0.2, 0.3]
    val_acc = [0.1, 0.15, 0.25]
    
    plot_training_metrics(train_losses, val_losses, train_acc, val_acc, save_dir=checkpoint_dir)
    
    assert os.path.exists(os.path.join(checkpoint_dir, 'training_plot.png'))
    print("plot_training_metrics passed.")
    
    # Cleanup
    shutil.rmtree(checkpoint_dir)
    print("Cleanup done.")

if __name__ == "__main__":
    test_plot()
