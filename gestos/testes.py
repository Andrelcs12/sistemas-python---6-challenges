import math
import unittest
from collections import deque

from main import angulo, detectar_movimento, distancia, valor_estavel


class TestesDaLogica(unittest.TestCase):
    def test_distancia(self):
        self.assertEqual(distancia((0, 0), (3, 4)), 5)

    def test_angulo_de_dedo_reto(self):
        self.assertAlmostEqual(angulo((0, 0), (1, 0), (2, 0)), 180)

    def test_angulo_de_dedo_dobrado(self):
        self.assertAlmostEqual(angulo((1, 0), (0, 0), (0, 1)), 90)

    def test_contagem_exige_repeticao(self):
        self.assertIsNone(valor_estavel([1, 2]))
        self.assertEqual(valor_estavel([3, 2, 3, 3, 1]), 3)

    def test_movimento_para_direita(self):
        posicoes = deque([(1.00, 0.2, 0.5), (1.20, 0.4, 0.5), (1.35, 0.6, 0.52)])
        self.assertEqual(detectar_movimento(posicoes, 1.35, -10), "DIREITA")

    def test_movimento_vertical_nao_conta_como_lateral(self):
        posicoes = deque([(1.00, 0.4, 0.1), (1.20, 0.4, 0.3), (1.35, 0.4, 0.6)])
        self.assertIsNone(detectar_movimento(posicoes, 1.35, -10))

    def test_evitar_movimento_repetido(self):
        posicoes = deque([(1.00, 0.2, 0.5), (1.20, 0.4, 0.5), (1.35, 0.7, 0.5)])
        self.assertIsNone(detectar_movimento(posicoes, 1.35, 1.0))


if __name__ == "__main__":
    unittest.main()
