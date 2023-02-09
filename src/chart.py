import os
import matplotlib.pyplot as plt
import seaborn as sns


class Chart:

    def __init__(self):
        self.PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..')).replace('\\', '//')
        plt.style.use(self.PATH + '//utils//scientific.mplstyle')

    def set_labels(self, xlabel, ylabel):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def bar_plot_plt(self, name, x, y, title, xlabel, ylabel):
        """
        Function to plot bar chart in black and white and using matplotlib
        """
        plt.figure(figsize=(16, 6))
        plt.xticks(fontsize=15, rotation=45)
        plt.yticks(fontsize=15)
        self.set_labels(xlabel, ylabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.bar(x, y, color="black")
        plt.savefig(self.PATH + f'//plots//{name}')
        return plt.show()

    def hist_plot_plt(self, name, features, labels, colors, xlabel, ylabel):
        plt.hist(features, label=labels, bins=30, color=colors)
        self.set_labels(xlabel, ylabel)
        plt.legend()
        plt.savefig(self.PATH + f'//plots//{name}')
        return plt.show()

    def line_comparison(self, name, list_features, line_style, linewidths, labels, colors, xlabel, ylabel):
        for index, feat in enumerate(list_features):
            plt.plot(
                feat,
                linewidth=linewidths[index],
                linestyle=line_style[index],
                color=colors[index],
                label=labels[index],
            )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(self.PATH + f'//plots//{name}')
        return plt.show()

    def scatter_reg_plot(self, name, x, y, color, xlabel, ylabel, limitsx, limitsy):
        sns.regplot(x=x, y=y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(limitsx[0], limitsx[1])
        plt.ylim(limitsy[0], limitsy[1])
        plt.savefig(self.PATH + f'//plots//{name}')
        return plt.show()

    def scatter_plot(self, name, x, y, color, xlabel, ylabel, limitsx, limitsy):
        plt.scatter(x, y, color=color)
        self.set_labels(xlabel, ylabel)
        plt.xlim(limitsx[0], limitsx[1])
        plt.ylim(limitsy[0], limitsy[1])
        plt.savefig(self.PATH + f'//plots//{name}')
        return plt.show()

    def comparison_type_curve(self, name, well, wells, colors, labels, xlabel, ylabel):
        for index, w in enumerate(wells):
            plt.plot(w, color=colors[index], label=labels[index])
        plt.plot(well, color="green", label="Pozo candidato", linewidth=2)
        self.set_labels(xlabel, ylabel)
        plt.legend()
        plt.savefig(self.PATH + f'//plots//{name}')
        return plt.show()

    def comparison_gradients(self, name, wells, colors, labels, xlabel, ylabel):
        for index, w in enumerate(wells):
            plt.scatter(w[0], w[1], color=colors[index], label=labels[index], s=100)
        sns.lineplot(x=[0, 0.09], y=[1.08, 0.4])
        self.set_labels(xlabel, ylabel)
        plt.xlim(0, 0.25)
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(self.PATH + f'//plots//{name}')
        return plt.show()

    def plot_learning_curves(self,name, train_sizes, train_mean, train_std, test_mean, test_std):
        # Draw lines
        plt.plot(train_sizes, train_mean, 'o-', color='red',  label='MAE de Entrenamiento')
        plt.plot(train_sizes, test_mean, 'o-', color='green', label='MAE de la validación cruzada')

        # Draw bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha = 0.1, color = 'r') # Alpha controls band transparency.
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha = 0.1, color = 'g')

        # Create plot
        plt.xlabel('Tamaño del set de entrenamiento')
        plt.ylabel('Error absolute medio')
        plt.legend(loc='best')
        plt.savefig(self.PATH + f'//plots//{name}')
        return plt.show()
