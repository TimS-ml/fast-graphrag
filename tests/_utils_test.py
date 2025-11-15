"""Unit tests for GraphRAG utility functions.

This module tests various utility functions including:
- Event loop management for asyncio operations
- Score extraction and sorting from sparse matrices
- CSR matrix creation from index lists
"""
# tests/test_utils.py

import asyncio
import threading
import unittest
from typing import List

import numpy as np
from scipy.sparse import csr_matrix

from fast_graphrag._utils import csr_from_indices_list, extract_sorted_scores, get_event_loop


class TestGetEventLoop(unittest.TestCase):
    """Test suite for event loop utility functions."""
    def test_get_existing_event_loop(self):
        """Test retrieval of an existing event loop in the current thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.assertEqual(get_event_loop(), loop)
        loop.close()

    def test_get_event_loop_in_sub_thread(self):
        """Test creation of a new event loop in a sub-thread.

        Verifies that get_event_loop() creates a new event loop
        when called from a thread without an existing loop.
        """
        def target():
            loop = get_event_loop()
            self.assertIsInstance(loop, asyncio.AbstractEventLoop)
            loop.close()

        thread = threading.Thread(target=target)
        thread.start()
        thread.join()


class TestExtractSortedScores(unittest.TestCase):
    """Test suite for extracting and sorting scores from sparse matrices."""
    def test_non_zero_elements(self):
        """Test extraction and sorting of non-zero elements.

        Verifies that scores are sorted in descending order with corresponding indices.
        """
        row_vector = csr_matrix([[0, 0.1, 0, 0.7, 0.5, 0]])
        indices, scores = extract_sorted_scores(row_vector)
        # Should return indices in descending order by score: 3 (0.7), 4 (0.5), 1 (0.1)
        np.testing.assert_array_equal(indices, np.array([3, 4, 1]))
        np.testing.assert_array_equal(scores, np.array([0.7, 0.5, 0.1]))

    def test_empty(self):
        """Test extraction from an empty matrix."""
        row_vector = csr_matrix((0, 0))
        indices, scores = extract_sorted_scores(row_vector)
        np.testing.assert_array_equal(indices, np.array([], dtype=np.int64))
        np.testing.assert_array_equal(scores, np.array([], dtype=np.float32))

    def test_empty_row_vector(self):
        """Test extraction from an empty row vector."""
        row_vector = csr_matrix([[]])
        indices, scores = extract_sorted_scores(row_vector)
        np.testing.assert_array_equal(indices, np.array([], dtype=np.int64))
        np.testing.assert_array_equal(scores, np.array([], dtype=np.float32))

    def test_single_element(self):
        """Test extraction from a single-element matrix."""
        row_vector = csr_matrix([[0.5]])
        indices, scores = extract_sorted_scores(row_vector)
        np.testing.assert_array_equal(indices, np.array([0]))
        np.testing.assert_array_equal(scores, np.array([0.5]))

    def test_all_zero_elements(self):
        """Test that all-zero elements return empty arrays."""
        row_vector = csr_matrix([[0, 0, 0, 0, 0]])
        indices, scores = extract_sorted_scores(row_vector)
        np.testing.assert_array_equal(indices, np.array([], dtype=np.int64))
        np.testing.assert_array_equal(scores, np.array([], dtype=np.float32))

    def test_duplicate_elements(self):
        """Test extraction with duplicate score values.

        When scores are equal, the order of indices may vary.
        This test accepts either valid ordering.
        """
        row_vector = csr_matrix([[0, 0.1, 0, 0.7, 0.5, 0.7]])
        indices, scores = extract_sorted_scores(row_vector)
        # Two possible orderings for indices with score 0.7 (indices 3 and 5)
        expected_indices_1 = np.array([5, 3, 4, 1])
        expected_indices_2 = np.array([3, 5, 4, 1])
        self.assertTrue(
            np.array_equal(indices, expected_indices_1) or np.array_equal(indices, expected_indices_2),
            f"indices {indices} do not match either {expected_indices_1} or {expected_indices_2}"
        )
        np.testing.assert_array_equal(scores, np.array([0.7, 0.7, 0.5, 0.1]))


class TestCsrFromListOfLists(unittest.TestCase):
    """Test suite for creating CSR matrices from lists of indices."""
    def test_repeated_elements(self):
        """Test CSR matrix creation with repeated indices.

        Verifies that duplicate indices in the same row are counted.
        """
        data: List[List[int]] = [[0, 0], [], []]
        expected_matrix = csr_matrix(([1, 1, 0], ([0, 0, 0], [0, 0, 0])), shape=(3, 3))
        result_matrix = csr_from_indices_list(data, shape=(3, 3))
        np.testing.assert_array_equal(result_matrix.toarray(), expected_matrix.toarray())

    def test_non_zero_elements(self):
        """Test CSR matrix creation with various non-zero indices.

        Creates a binary matrix where 1 indicates presence of an index.
        """
        data = [[0, 1, 2], [2, 3], [0, 3]]
        expected_matrix = csr_matrix([[1, 1, 1, 0, 0], [0, 0, 1, 1, 0], [1, 0, 0, 1, 0]], shape=(3, 5))
        result_matrix = csr_from_indices_list(data, shape=(3, 5))
        np.testing.assert_array_equal(result_matrix.toarray(), expected_matrix.toarray())

    def test_empty_list_of_lists(self):
        """Test CSR matrix creation from an empty list."""
        data: List[List[int]] = []
        expected_matrix = csr_matrix((0, 0))
        result_matrix = csr_from_indices_list(data, shape=(0, 0))
        np.testing.assert_array_equal(result_matrix.toarray(), expected_matrix.toarray())

    def test_empty_list_of_lists_with_unempty_shape(self):
        """Test CSR matrix creation with empty data but non-zero shape.

        Results in a matrix of zeros with the specified shape.
        """
        data: List[List[int]] = []
        expected_matrix = csr_matrix((1, 1))
        result_matrix = csr_from_indices_list(data, shape=(1, 1))
        np.testing.assert_array_equal(result_matrix.toarray(), expected_matrix.toarray())

    def test_list_with_empty_sublists(self):
        """Test CSR matrix creation with empty sublists.

        Results in a matrix with rows but no columns.
        """
        data: List[List[int]] = [[], [], []]
        expected_matrix = csr_matrix((3, 0))
        result_matrix = csr_from_indices_list(data, shape=(3, 0))
        np.testing.assert_array_equal(result_matrix.toarray(), expected_matrix.toarray())


if __name__ == "__main__":
    unittest.main()
