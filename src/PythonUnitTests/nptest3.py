import math
import numpy as np

einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)

class nptest3(object):
    """description of class"""

    @staticmethod
    def _compute_size_by_dict(indices, idx_dict):
        """
        Computes the product of the elements in indices based on the dictionary
        idx_dict.

        Parameters
        ----------
        indices : iterable
            Indices to base the product on.
        idx_dict : dictionary
            Dictionary of index sizes

        Returns
        -------
        ret : int
            The resulting product.

        Examples
        --------
        >>> _compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
        90

        """
        ret = 1
        for i in indices:
            ret *= idx_dict[i]
        return ret

    @staticmethod
    def _find_contraction(positions, input_sets, output_set):
        """
        Finds the contraction for a given set of input and output sets.

        Parameters
        ----------
        positions : iterable
            Integer positions of terms used in the contraction.
        input_sets : list
            List of sets that represent the lhs side of the einsum subscript
        output_set : set
            Set that represents the rhs side of the overall einsum subscript

        Returns
        -------
        new_result : set
            The indices of the resulting contraction
        remaining : list
            List of sets that have not been contracted, the new set is appended to
            the end of this list
        idx_removed : set
            Indices removed from the entire contraction
        idx_contraction : set
            The indices used in the current contraction

        Examples
        --------

        # A simple dot product test case
        >>> pos = (0, 1)
        >>> isets = [set('ab'), set('bc')]
        >>> oset = set('ac')
        >>> _find_contraction(pos, isets, oset)
        ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})

        # A more complex case with additional terms in the contraction
        >>> pos = (0, 2)
        >>> isets = [set('abd'), set('ac'), set('bdc')]
        >>> oset = set('ac')
        >>> _find_contraction(pos, isets, oset)
        ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})
        """

        idx_contract = set()
        idx_remain = output_set.copy()
        remaining = []
        for ind, value in enumerate(input_sets):
            if ind in positions:
                idx_contract |= value
            else:
                remaining.append(value)
                idx_remain |= value

        new_result = idx_remain & idx_contract
        idx_removed = (idx_contract - new_result)
        remaining.append(new_result)

        return (new_result, remaining, idx_removed, idx_contract)

    @staticmethod
    def _optimal_path(input_sets, output_set, idx_dict, memory_limit):
        """
        Computes all possible pair contractions, sieves the results based
        on ``memory_limit`` and returns the lowest cost path. This algorithm
        scales factorial with respect to the elements in the list ``input_sets``.

        Parameters
        ----------
        input_sets : list
            List of sets that represent the lhs side of the einsum subscript
        output_set : set
            Set that represents the rhs side of the overall einsum subscript
        idx_dict : dictionary
            Dictionary of index sizes
        memory_limit : int
            The maximum number of elements in a temporary array

        Returns
        -------
        path : list
            The optimal contraction order within the memory limit constraint.

        Examples
        --------
        >>> isets = [set('abd'), set('ac'), set('bdc')]
        >>> oset = set('')
        >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
        >>> _path__optimal_path(isets, oset, idx_sizes, 5000)
        [(0, 2), (0, 1)]
        """

        full_results = [(0, [], input_sets)]
        for iteration in range(len(input_sets) - 1):
            iter_results = []

            # Compute all unique pairs
            comb_iter = []
            for x in range(len(input_sets) - iteration):
                for y in range(x + 1, len(input_sets) - iteration):
                    comb_iter.append((x, y))

            for curr in full_results:
                cost, positions, remaining = curr
                for con in comb_iter:

                    # Find the contraction
                    cont = nptest3._find_contraction(con, remaining, output_set)
                    new_result, new_input_sets, idx_removed, idx_contract = cont

                    # Sieve the results based on memory_limit
                    new_size = nptest3._compute_size_by_dict(new_result, idx_dict)
                    if new_size > memory_limit:
                        continue

                    # Find cost
                    new_cost = nptest3._compute_size_by_dict(idx_contract, idx_dict)
                    if idx_removed:
                        new_cost *= 2

                    # Build (total_cost, positions, indices_remaining)
                    new_cost += cost
                    new_pos = positions + [con]
                    iter_results.append((new_cost, new_pos, new_input_sets))

            # Update combinatorial list, if we did not find anything return best
            # path + remaining contractions
            if iter_results:
                full_results = iter_results
            else:
                path = min(full_results, key=lambda x: x[0])[1]
                path += [tuple(range(len(input_sets) - iteration))]
                return path

        # If we have not found anything return single einsum contraction
        if len(full_results) == 0:
            return [tuple(range(len(input_sets)))]

        path = min(full_results, key=lambda x: x[0])[1]
        return path

    @staticmethod
    def _greedy_path(input_sets, output_set, idx_dict, memory_limit):
        """
        Finds the path by contracting the best pair until the input list is
        exhausted. The best pair is found by minimizing the tuple
        ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing
        matrix multiplication or inner product operations, then Hadamard like
        operations, and finally outer operations. Outer products are limited by
        ``memory_limit``. This algorithm scales cubically with respect to the
        number of elements in the list ``input_sets``.

        Parameters
        ----------
        input_sets : list
            List of sets that represent the lhs side of the einsum subscript
        output_set : set
            Set that represents the rhs side of the overall einsum subscript
        idx_dict : dictionary
            Dictionary of index sizes
        memory_limit_limit : int
            The maximum number of elements in a temporary array

        Returns
        -------
        path : list
            The greedy contraction order within the memory limit constraint.

        Examples
        --------
        >>> isets = [set('abd'), set('ac'), set('bdc')]
        >>> oset = set('')
        >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
        >>> _path__greedy_path(isets, oset, idx_sizes, 5000)
        [(0, 2), (0, 1)]
        """

        if len(input_sets) == 1:
            return [(0,)]

        path = []
        for iteration in range(len(input_sets) - 1):
            iteration_results = []
            comb_iter = []

            # Compute all unique pairs
            for x in range(len(input_sets)):
                for y in range(x + 1, len(input_sets)):
                    comb_iter.append((x, y))

            for positions in comb_iter:

                # Find the contraction
                contract = nptest3._find_contraction(positions, input_sets, output_set)
                idx_result, new_input_sets, idx_removed, idx_contract = contract

                # Sieve the results based on memory_limit
                if nptest3._compute_size_by_dict(idx_result, idx_dict) > memory_limit:
                    continue

                # Build sort tuple
                removed_size = nptest3._compute_size_by_dict(idx_removed, idx_dict)
                cost = nptest3._compute_size_by_dict(idx_contract, idx_dict)
                sort = (-removed_size, cost)

                # Add contraction to possible choices
                iteration_results.append([sort, positions, new_input_sets])

            # If we did not find a new contraction contract remaining
            if len(iteration_results) == 0:
                path.append(tuple(range(len(input_sets))))
                break

            # Sort based on first index
            best = min(iteration_results, key=lambda x: x[0])
            path.append(best[1])
            input_sets = best[2]

        return path

    @staticmethod
    def _can_dot(inputs, result, idx_removed):
        """
        Checks if we can use BLAS (np.tensordot) call and its beneficial to do so.

        Parameters
        ----------
        inputs : list of str
            Specifies the subscripts for summation.
        result : str
            Resulting summation.
        idx_removed : set
            Indices that are removed in the summation


        Returns
        -------
        type : bool
            Returns true if BLAS should and can be used, else False

        Notes
        -----
        If the operations is BLAS level 1 or 2 and is not already aligned
        we default back to einsum as the memory movement to copy is more
        costly than the operation itself.


        Examples
        --------

        # Standard GEMM operation
        >>> _can_dot(['ij', 'jk'], 'ik', set('j'))
        True

        # Can use the standard BLAS, but requires odd data movement
        >>> _can_dot(['ijj', 'jk'], 'ik', set('j'))
        False

        # DDOT where the memory is not aligned
        >>> _can_dot(['ijk', 'ikj'], '', set('ijk'))
        False

        """

        # All `dot` calls remove indices
        if len(idx_removed) == 0:
            return False

        # BLAS can only handle two operands
        if len(inputs) != 2:
            return False

        # Build a few temporaries
        input_left, input_right = inputs
        set_left = set(input_left)
        set_right = set(input_right)
        keep_left = set_left - idx_removed
        keep_right = set_right - idx_removed
        rs = len(idx_removed)

        # Indices must overlap between the two operands
        if not len(set_left & set_right):
            return False

        # We cannot have duplicate indices ("ijj, jk -> ik")
        if (len(set_left) != len(input_left)) or (len(set_right) != len(input_right)):
            return False

        # Cannot handle partial inner ("ij, ji -> i")
        if len(keep_left & keep_right):
            return False

        # At this point we are a DOT, GEMV, or GEMM operation

        # Handle inner products

        # DDOT with aligned data
        if input_left == input_right:
            return True

        # DDOT without aligned data (better to use einsum)
        if set_left == set_right:
            return False

        # Handle the 4 possible (aligned) GEMV or GEMM cases

        # GEMM or GEMV no transpose
        if input_left[-rs:] == input_right[:rs]:
            return True

        # GEMM or GEMV transpose both
        if input_left[:rs] == input_right[-rs:]:
            return True

        # GEMM or GEMV transpose right
        if input_left[-rs:] == input_right[-rs:]:
            return True

        # GEMM or GEMV transpose left
        if input_left[:rs] == input_right[:rs]:
            return True

        # Einsum is faster than GEMV if we have to copy data
        if not keep_left or not keep_right:
            return False

        # We are a matrix-matrix product, but we need to copy data
        return True

    @staticmethod
    def _parse_einsum_input(operands):
        """
        A reproduction of einsum c side einsum parsing in python.

        Returns
        -------
        input_strings : str
            Parsed input strings
        output_string : str
            Parsed output string
        operands : list of array_like
            The operands to use in the numpy contraction

        Examples
        --------
        The operand list is simplified to reduce printing:

        >>> a = np.random.rand(4, 4)
        >>> b = np.random.rand(4, 4, 4)
        >>> __parse_einsum_input(('...a,...a->...', a, b))
        ('za,xza', 'xz', [a, b])

        >>> __parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
        ('za,xza', 'xz', [a, b])
        """

        if len(operands) == 0:
            raise ValueError("No input operands")

        if isinstance(operands[0], basestring):
            subscripts = operands[0].replace(" ", "")
            operands = [np.asanyarray(v) for v in operands[1:]]

            # Ensure all characters are valid
            for s in subscripts:
                if s in '.,->':
                    continue
                if s not in einsum_symbols:
                    raise ValueError("Character %s is not a valid symbol." % s)

        else:
            tmp_operands = list(operands)
            operand_list = []
            subscript_list = []
            for p in range(len(operands) // 2):
                operand_list.append(tmp_operands.pop(0))
                subscript_list.append(tmp_operands.pop(0))

            output_list = tmp_operands[-1] if len(tmp_operands) else None
            operands = [np.asanyarray(v) for v in operand_list]
            subscripts = ""
            last = len(subscript_list) - 1
            for num, sub in enumerate(subscript_list):
                for s in sub:
                    if s is Ellipsis:
                        subscripts += "..."
                    elif isinstance(s, int):
                        subscripts += einsum_symbols[s]
                    else:
                        raise TypeError("For this input type lists must contain "
                                        "either int or Ellipsis")
                if num != last:
                    subscripts += ","

            if output_list is not None:
                subscripts += "->"
                for s in output_list:
                    if s is Ellipsis:
                        subscripts += "..."
                    elif isinstance(s, int):
                        subscripts += einsum_symbols[s]
                    else:
                        raise TypeError("For this input type lists must contain "
                                        "either int or Ellipsis")
        # Check for proper "->"
        if ("-" in subscripts) or (">" in subscripts):
            invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
            if invalid or (subscripts.count("->") != 1):
                raise ValueError("Subscripts can only contain one '->'.")

        # Parse ellipses
        if "." in subscripts:
            used = subscripts.replace(".", "").replace(",", "").replace("->", "")
            unused = list(einsum_symbols_set - set(used))
            ellipse_inds = "".join(unused)
            longest = 0

            if "->" in subscripts:
                input_tmp, output_sub = subscripts.split("->")
                split_subscripts = input_tmp.split(",")
                out_sub = True
            else:
                split_subscripts = subscripts.split(',')
                out_sub = False

            for num, sub in enumerate(split_subscripts):
                if "." in sub:
                    if (sub.count(".") != 3) or (sub.count("...") != 1):
                        raise ValueError("Invalid Ellipses.")

                    # Take into account numerical values
                    if operands[num].shape == ():
                        ellipse_count = 0
                    else:
                        ellipse_count = max(operands[num].ndim, 1)
                        ellipse_count -= (len(sub) - 3)

                    if ellipse_count > longest:
                        longest = ellipse_count

                    if ellipse_count < 0:
                        raise ValueError("Ellipses lengths do not match.")
                    elif ellipse_count == 0:
                        split_subscripts[num] = sub.replace('...', '')
                    else:
                        rep_inds = ellipse_inds[-ellipse_count:]
                        split_subscripts[num] = sub.replace('...', rep_inds)

            subscripts = ",".join(split_subscripts)
            if longest == 0:
                out_ellipse = ""
            else:
                out_ellipse = ellipse_inds[-longest:]

            if out_sub:
                subscripts += "->" + output_sub.replace("...", out_ellipse)
            else:
                # Special care for outputless ellipses
                output_subscript = ""
                tmp_subscripts = subscripts.replace(",", "")
                for s in sorted(set(tmp_subscripts)):
                    if s not in (einsum_symbols):
                        raise ValueError("Character %s is not a valid symbol." % s)
                    if tmp_subscripts.count(s) == 1:
                        output_subscript += s
                normal_inds = ''.join(sorted(set(output_subscript) -
                                             set(out_ellipse)))

                subscripts += "->" + out_ellipse + normal_inds

        # Build output string if does not exist
        if "->" in subscripts:
            input_subscripts, output_subscript = subscripts.split("->")
        else:
            input_subscripts = subscripts
            # Build output subscripts
            tmp_subscripts = subscripts.replace(",", "")
            output_subscript = ""
            for s in sorted(set(tmp_subscripts)):
                if s not in einsum_symbols:
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s

        # Make sure output subscripts are in the input
        for char in output_subscript:
            if char not in input_subscripts:
                raise ValueError("Output character %s did not appear in the input"
                                 % char)

        # Make sure number operands is equivalent to the number of terms
        if len(input_subscripts.split(',')) != len(operands):
            raise ValueError("Number of einsum subscripts must be equal to the "
                             "number of operands.")

        return (input_subscripts, output_subscript, operands)


    @staticmethod
    def einsum_path(*operands, **kwargs):
        """
        einsum_path(subscripts, *operands, optimize='greedy')

        Evaluates the lowest cost contraction order for an einsum expression by
        considering the creation of intermediate arrays.

        Parameters
        ----------
        subscripts : str
            Specifies the subscripts for summation.
        *operands : list of array_like
            These are the arrays for the operation.
        optimize : {bool, list, tuple, 'greedy', 'optimal'}
            Choose the type of path. If a tuple is provided, the second argument is
            assumed to be the maximum intermediate size created. If only a single
            argument is provided the largest input or output array size is used
            as a maximum intermediate size.

            * if a list is given that starts with ``einsum_path``, uses this as the
              contraction path
            * if False no optimization is taken
            * if True defaults to the 'greedy' algorithm
            * 'optimal' An algorithm that combinatorially explores all possible
              ways of contracting the listed tensors and choosest the least costly
              path. Scales exponentially with the number of terms in the
              contraction.
            * 'greedy' An algorithm that chooses the best pair contraction
              at each step. Effectively, this algorithm searches the largest inner,
              Hadamard, and then outer products at each step. Scales cubically with
              the number of terms in the contraction. Equivalent to the 'optimal'
              path for most contractions.

            Default is 'greedy'.

        Returns
        -------
        path : list of tuples
            A list representation of the einsum path.
        string_repr : str
            A printable representation of the einsum path.

        Notes
        -----
        The resulting path indicates which terms of the input contraction should be
        contracted first, the result of this contraction is then appended to the
        end of the contraction list. This list can then be iterated over until all
        intermediate contractions are complete.

        See Also
        --------
        einsum, linalg.multi_dot

        Examples
        --------

        We can begin with a chain dot example. In this case, it is optimal to
        contract the ``b`` and ``c`` tensors first as represented by the first
        element of the path ``(1, 2)``. The resulting tensor is added to the end
        of the contraction and the remaining contraction ``(0, 1)`` is then
        completed.

        >>> a = np.random.rand(2, 2)
        >>> b = np.random.rand(2, 5)
        >>> c = np.random.rand(5, 2)
        >>> path_info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
        >>> print(path_info[0])
        ['einsum_path', (1, 2), (0, 1)]
        >>> print(path_info[1])
          Complete contraction:  ij,jk,kl->il
                 Naive scaling:  4
             Optimized scaling:  3
              Naive FLOP count:  1.600e+02
          Optimized FLOP count:  5.600e+01
           Theoretical speedup:  2.857
          Largest intermediate:  4.000e+00 elements
        -------------------------------------------------------------------------
        scaling                  current                                remaining
        -------------------------------------------------------------------------
           3                   kl,jk->jl                                ij,jl->il
           3                   jl,ij->il                                   il->il


        A more complex index transformation example.

        >>> I = np.random.rand(10, 10, 10, 10)
        >>> C = np.random.rand(10, 10)
        >>> path_info = np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C,
                                       optimize='greedy')

        >>> print(path_info[0])
        ['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)]
        >>> print(path_info[1])
          Complete contraction:  ea,fb,abcd,gc,hd->efgh
                 Naive scaling:  8
             Optimized scaling:  5
              Naive FLOP count:  8.000e+08
          Optimized FLOP count:  8.000e+05
           Theoretical speedup:  1000.000
          Largest intermediate:  1.000e+04 elements
        --------------------------------------------------------------------------
        scaling                  current                                remaining
        --------------------------------------------------------------------------
           5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
           5               bcde,fb->cdef                         gc,hd,cdef->efgh
           5               cdef,gc->defg                            hd,defg->efgh
           5               defg,hd->efgh                               efgh->efgh
        """

        # Make sure all keywords are valid
        valid_contract_kwargs = ['optimize', 'einsum_call']
        unknown_kwargs = [k for (k, v) in kwargs.items() if k
                          not in valid_contract_kwargs]
        if len(unknown_kwargs):
            raise TypeError("Did not understand the following kwargs:"
                            " %s" % unknown_kwargs)

        # Figure out what the path really is
        path_type = kwargs.pop('optimize', True)
        if path_type is True:
            path_type = 'greedy'
        if path_type is None:
            path_type = False

        memory_limit = None

        # No optimization or a named path algorithm
        if (path_type is False) or isinstance(path_type, basestring):
            pass

        # Given an explicit path
        elif len(path_type) and (path_type[0] == 'einsum_path'):
            pass

        # Path tuple with memory limit
        elif ((len(path_type) == 2) and isinstance(path_type[0], basestring) and
                isinstance(path_type[1], (int, float))):
            memory_limit = int(path_type[1])
            path_type = path_type[0]

        else:
            raise TypeError("Did not understand the path: %s" % str(path_type))

        # Hidden option, only einsum should call this
        einsum_call_arg = kwargs.pop("einsum_call", False)

        # Python side parsing
        input_subscripts, output_subscript, operands = nptest3._parse_einsum_input(operands)
        subscripts = input_subscripts + '->' + output_subscript

        # Build a few useful list and sets
        input_list = input_subscripts.split(',')
        input_sets = [set(x) for x in input_list]
        output_set = set(output_subscript)
        indices = set(input_subscripts.replace(',', ''))

        # Get length of each unique dimension and ensure all dimensions are correct
        dimension_dict = {}
        for tnum, term in enumerate(input_list):
            sh = operands[tnum].shape
            if len(sh) != len(term):
                raise ValueError("Einstein sum subscript %s does not contain the "
                                 "correct number of indices for operand %d."
                                 % (input_subscripts[tnum], tnum))
            for cnum, char in enumerate(term):
                dim = sh[cnum]
                if char in dimension_dict.keys():
                    # For broadcasting cases we always want the largest dim size
                    if dimension_dict[char] == 1:
                        dimension_dict[char] = dim
                    elif dim not in (1, dimension_dict[char]):
                        raise ValueError("Size of label '%s' for operand %d (%d) "
                                         "does not match previous terms (%d)."
                                         % (char, tnum, dimension_dict[char], dim))
                else:
                    dimension_dict[char] = dim

        # Compute size of each input array plus the output array
        size_list = []
        for term in input_list + [output_subscript]:
            size_list.append(nptest3._compute_size_by_dict(term, dimension_dict))
        max_size = max(size_list)

        if memory_limit is None:
            memory_arg = max_size
        else:
            memory_arg = memory_limit

        # Compute naive cost
        # This isn't quite right, need to look into exactly how einsum does this
        naive_cost = nptest3._compute_size_by_dict(indices, dimension_dict)
        indices_in_input = input_subscripts.replace(',', '')
        mult = max(len(input_list) - 1, 1)
        if (len(indices_in_input) - len(set(indices_in_input))):
            mult *= 2
        naive_cost *= mult

        # Compute the path
        if (path_type is False) or (len(input_list) in [1, 2]) or (indices == output_set):
            # Nothing to be optimized, leave it to einsum
            path = [tuple(range(len(input_list)))]
        elif path_type == "greedy":
            # Maximum memory should be at most out_size for this algorithm
            memory_arg = min(memory_arg, max_size)
            path = nptest3._greedy_path(input_sets, output_set, dimension_dict, memory_arg)
        elif path_type == "optimal":
            path = nptest3._optimal_path(input_sets, output_set, dimension_dict, memory_arg)
        elif path_type[0] == 'einsum_path':
            path = path_type[1:]
        else:
            raise KeyError("Path name %s not found", path_type)

        cost_list, scale_list, size_list, contraction_list = [], [], [], []

        # Build contraction tuple (positions, gemm, einsum_str, remaining)
        for cnum, contract_inds in enumerate(path):
            # Make sure we remove inds from right to left
            contract_inds = tuple(sorted(list(contract_inds), reverse=True))

            contract = nptest3._find_contraction(contract_inds, input_sets, output_set)
            out_inds, input_sets, idx_removed, idx_contract = contract

            cost = nptest3._compute_size_by_dict(idx_contract, dimension_dict)
            if idx_removed:
                cost *= 2
            cost_list.append(cost)
            scale_list.append(len(idx_contract))
            size_list.append(nptest3._compute_size_by_dict(out_inds, dimension_dict))

            tmp_inputs = []
            for x in contract_inds:
                tmp_inputs.append(input_list.pop(x))

            do_blas = nptest3._can_dot(tmp_inputs, out_inds, idx_removed)

            # Last contraction
            if (cnum - len(path)) == -1:
                idx_result = output_subscript
            else:
                sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
                idx_result = "".join([x[1] for x in sorted(sort_result)])

            input_list.append(idx_result)
            einsum_str = ",".join(tmp_inputs) + "->" + idx_result

            contraction = (contract_inds, idx_removed, einsum_str, input_list[:], do_blas)
            contraction_list.append(contraction)

        opt_cost = sum(cost_list) + 1

        if einsum_call_arg:
            return (operands, contraction_list)

        # Return the path along with a nice string representation
        overall_contraction = input_subscripts + "->" + output_subscript
        header = ("scaling", "current", "remaining")

        speedup = naive_cost / opt_cost
        max_i = max(size_list)

        path_print  = "  Complete contraction:  %s\n" % overall_contraction
        path_print += "         Naive scaling:  %d\n" % len(indices)
        path_print += "     Optimized scaling:  %d\n" % max(scale_list)
        path_print += "      Naive FLOP count:  %.3e\n" % naive_cost
        path_print += "  Optimized FLOP count:  %.3e\n" % opt_cost
        path_print += "   Theoretical speedup:  %3.3f\n" % speedup
        path_print += "  Largest intermediate:  %.3e elements\n" % max_i
        path_print += "-" * 74 + "\n"
        path_print += "%6s %24s %40s\n" % header
        path_print += "-" * 74

        for n, contraction in enumerate(contraction_list):
            inds, idx_rm, einsum_str, remaining, blas = contraction
            remaining_str = ",".join(remaining) + "->" + output_subscript
            path_run = (scale_list[n], einsum_str, remaining_str)
            path_print += "\n%4d    %24s %40s" % path_run

        path = ['einsum_path'] + path
        return (path, path_print)


