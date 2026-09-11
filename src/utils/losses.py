def penalize_smaller_loss_julia(confidence):
    return """
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}

        # get predicted values for the current tree
        prediction, flag = eval_tree_array(tree, dataset.X, options)

        # 'flag' == false means that evaluating the tree caused an error
        if !flag
            return L(Inf)
        end

        result = 0.0
        coverage = 0.0
        coverage_penalty = 100.0
        target_coverage = %.2f

        # instead of just having a sum of squared means, we penalize more heavily
        # samples for which the predictions are inferior to 'y' (here the difference
        # between the true value and the predicted value)
        for i in 1:length(dataset.y)
            if (prediction[i] < dataset.y[i])
                result += 10 * (prediction[i] - dataset.y[i])^2
            else
                result += (prediction[i] - dataset.y[i])^2
                coverage += 1
            end
        end

        if ((coverage / dataset.n) < target_coverage)
            # penalty is equal to the difference between complete coverage and current result * weight
            result += (target_coverage - coverage/dataset.n) * dataset.n * coverage_penalty
        end

        return result / dataset.n
    end
    """ % confidence


def bin_crossfit_loss_julia(confidence):
    return """
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        # tree predicts log(sigma); dataset.y holds the raw OOB residual
        log_sigma, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        sigma = exp.(log_sigma)
        if any(sigma .<= zero(T)) || !all(isfinite.(sigma))
            return L(Inf)
        end

        n = dataset.n
        if n < 8
            # too few points for a 2-fold x 4-bin split to mean anything
            return L(Inf)
        end

        target_coverage = L(%.2f)
        alpha = one(L) - target_coverage
        n_bins = 4
        lambda_cov = L(1.0)

        delta = abs.(dataset.y) ./ sigma

        sigma_rank = sortperm(sigma)
        bin_id = Vector{Int}(undef, n)
        for (rank, idx) in enumerate(sigma_rank)
            bin_id[idx] = clamp(ceil(Int, rank * n_bins / n), 1, n_bins)
        end

        fold_perm = sortperm(rand(n))
        half = n ÷ 2
        fold_A = fold_perm[1:half]
        fold_B = fold_perm[half+1:end]

        function fold_pass(cal_idx, eval_idx)
            width_sum = zero(L)
            width_count = 0
            cov_sum = zeros(L, n_bins)
            cov_count = zeros(Int, n_bins)
            for b in 1:n_bins
                cal_in_bin = [i for i in cal_idx if bin_id[i] == b]
                eval_in_bin = [i for i in eval_idx if bin_id[i] == b]
                if isempty(cal_in_bin) || isempty(eval_in_bin)
                    continue
                end
                cal_scores = sort(delta[cal_in_bin])
                m = length(cal_scores)
                q_pos = clamp(ceil(Int, (one(L) - alpha) * m), 1, m)
                q_hat = L(cal_scores[q_pos])

                for i in eval_in_bin
                    width_sum += q_hat * L(sigma[i])
                    width_count += 1
                    if L(delta[i]) <= q_hat
                        cov_sum[b] += one(L)
                    end
                    cov_count[b] += 1
                end
            end
            return width_sum, width_count, cov_sum, cov_count
        end

        width_sum_A, width_count_A, cov_sum_A, cov_count_A = fold_pass(fold_A, fold_B)
        width_sum_B, width_count_B, cov_sum_B, cov_count_B = fold_pass(fold_B, fold_A)

        if width_count_A == 0 || width_count_B == 0
            return L(Inf)
        end

        # ell_w: mean interval width, averaged over both cross-fit directions
        ell_w = (width_sum_A / width_count_A + width_sum_B / width_count_B) / L(2.0)

        # ell_cov per bin, averaged over both cross-fit directions, then
        # penalized by its squared deviation from the target coverage
        coverage_penalty = zero(L)
        n_valid_bins = 0
        for b in 1:n_bins
            if cov_count_A[b] == 0 || cov_count_B[b] == 0
                continue
            end
            ell_cov_A = cov_sum_A[b] / cov_count_A[b]
            ell_cov_B = cov_sum_B[b] / cov_count_B[b]
            mean_cov = (ell_cov_A + ell_cov_B) / L(2.0)
            coverage_penalty += (mean_cov - target_coverage)^2
            n_valid_bins += 1
        end
        if n_valid_bins == 0
            return L(Inf)
        end

        return ell_w + lambda_cov * coverage_penalty
    end
    """ % confidence