{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "standard_error.py",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/kumagaimasahito/OpenJij/blob/feature%2Ferror_bar/openjij/utils/standard_error.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xqK9wjtrCw1P",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np\n",
        "import scipy as sp\n",
        "from scipy import stats\n",
        "\n",
        "\n",
        "def standard_error(solver, time_list, iteration_list, args={}):\n",
        "    \"\"\"Calculate 'Standard Error' with iteration\n",
        "    Args:\n",
        "        solver (callable): returns openjij.Response, and solver has arguments 'time' and '**args'\n",
        "        time_list (list):\n",
        "        iteration_list (list):\n",
        "        args (dict): Arguments for solver.\n",
        "\n",
        "    Returns:\n",
        "        dict: {\n",
        "                \"se\": list of standard error at each iteration\n",
        "                \"info\" (dict): Parameter information for the benchmark\n",
        "            }\n",
        "    \"\"\"\n",
        "\n",
        "    se_list = []\n",
        "    response_list = []\n",
        "\n",
        "    i = 0\n",
        "    for iteration in iteration_list:\n",
        "        for time in time_list:\n",
        "            response = solver(time, **args)\n",
        "            response_list.append(response)\n",
        "        se_list[i] = sp.std(response_list[i].energies, ddof=1)\n",
        "        i = i + 1\n",
        "\n",
        "\n",
        "    return {\n",
        "        \"se\": standard_error,\n",
        "        \"info\":{\n",
        "            \"iteration_list\": iteration_list,\n",
        "            }\n",
        "        }"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}