import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_filters_activations(image, net):
    conv1_feature_maps = net.conv1(image)
    conv1_filter = net.conv1.weight

    conv2_feature_maps = net.conv2(net.fconv1(image))
    conv2_filter = net.conv2.weight

    for (activation_output, square) in [(conv1_feature_maps, 4), (conv2_feature_maps, 6)]:
        position = 1
        for feature in activation_output:
            ax = plt.subplot(square, square, position)
            ax.set_xticks([])
            ax.set_yticks([])

            plt.imshow(feature.cpu().detach().numpy(), cmap='gray')
            position += 1
        plt.show()

    for (filter_list, square) in [(conv1_filter, 4), (conv2_filter, 6)]:
        position = 1
        for filter in filter_list:
            ax = plt.subplot(square, square, position)
            ax.set_xticks([])
            ax.set_yticks([])

            plt.imshow(filter[0].cpu().detach().numpy(), cmap='gray')
            position += 1
        plt.show()


def visualize_pca(feature_list, images, labels, scatter_images, img_limit):
    for features in feature_list:
        transformed_data = PCA(n_components=2).fit_transform(features)
        plot_scatter(transformed_data, images, labels, scatter_images, img_limit)


def visualize_tsne(feature_list, images, labels, scatter_images, img_limit):
    for features in feature_list:
        tsne_transformed = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3) \
            .fit_transform(features)
        plot_scatter(tsne_transformed, images, labels, scatter_images, img_limit)


def plot_scatter(transformed_data, images, labels, scatter_images, img_limit):
    if scatter_images:
        fig, ax = plt.subplots()
        minimum_x = math.inf
        maximum_x = - math.inf
        minimum_y = math.inf
        maximum_y = - math.inf

        for i, image in enumerate(images):
            minimum_x = transformed_data[i, 0] if transformed_data[i, 0] < minimum_x else minimum_x
            minimum_y = transformed_data[i, 1] if transformed_data[i, 1] < minimum_y else minimum_y

            maximum_x = transformed_data[i, 0] if transformed_data[i, 0] > maximum_x else maximum_x
            maximum_y = transformed_data[i, 1] if transformed_data[i, 1] > maximum_y else maximum_y

            ax.imshow(image[0], cmap='gray',
                      extent=(transformed_data[i, 0] - img_limit,
                              transformed_data[i, 0] + img_limit,
                              transformed_data[i, 1] - img_limit,
                              transformed_data[i, 1] + img_limit))

        ax.set_xlim([minimum_x - 1, maximum_x + 1])
        ax.set_ylim([minimum_y - 1, maximum_y + 1])
        plt.show()

    else:
        fig, ax = plt.subplots()
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels)
        fig.colorbar(scatter)
        plt.show()