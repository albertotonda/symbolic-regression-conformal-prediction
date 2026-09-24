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


def bin_crossfit_loss_julia(confidence, lambda_cov, seed=0):
    """Julia loss for the sigma-SR search: scores the normalized conformal
    predictor that gets deployed (intervals q * sigma(x), one global q), with
    2-fold cross-fitting so q is never calibrated on the points it scores.

    loss = mean half-width + lambda_cov * sum over 4 sigma-rank bins of
           (bin coverage - target coverage)^2

    The fold split is a fixed permutation (sorted hashes of (index, seed)),
    so every candidate equation is scored on the same split.
    """
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

        target_coverage = L(%.4f)
        alpha = one(L) - target_coverage
        n_bins = 4

        delta = abs.(dataset.y) ./ sigma

        # bins by sigma rank, only used to measure conditional coverage
        sigma_rank = sortperm(sigma)
        bin_id = Vector{Int}(undef, n)
        for (rank, idx) in enumerate(sigma_rank)
            bin_id[idx] = clamp(ceil(Int, rank * n_bins / n), 1, n_bins)
        end

        fold_perm = sortperm([hash((i, %d)) for i in 1:n])
        half = n ÷ 2
        fold_A = fold_perm[1:half]
        fold_B = fold_perm[half+1:end]

        function fold_pass(cal_idx, eval_idx)
            # one global conformal quantile from the calibration fold,
            # same finite-sample rank as crepes: ceil((1 - alpha)(m + 1))
            cal_scores = sort(delta[cal_idx])
            m = length(cal_scores)
            q_pos = clamp(ceil(Int, (one(L) - alpha) * (m + 1)), 1, m)
            q_hat = L(cal_scores[q_pos])

            width_sum = zero(L)
            cov_sum = zeros(L, n_bins)
            cov_count = zeros(Int, n_bins)
            for i in eval_idx
                width_sum += q_hat * L(sigma[i])
                b = bin_id[i]
                if L(delta[i]) <= q_hat
                    cov_sum[b] += one(L)
                end
                cov_count[b] += 1
            end
            return width_sum / length(eval_idx), cov_sum, cov_count
        end

        width_A, cov_sum_A, cov_count_A = fold_pass(fold_A, fold_B)
        width_B, cov_sum_B, cov_count_B = fold_pass(fold_B, fold_A)

        # ell_w: mean interval half-width, averaged over both cross-fit directions
        ell_w = (width_A + width_B) / L(2.0)

        # per-bin coverage, pooled over both directions, penalized by its
        # squared deviation from the target coverage
        coverage_penalty = zero(L)
        for b in 1:n_bins
            count = cov_count_A[b] + cov_count_B[b]
            if count == 0
                continue
            end
            bin_coverage = (cov_sum_A[b] + cov_sum_B[b]) / count
            coverage_penalty += (bin_coverage - target_coverage)^2
        end

        return ell_w + L(%.4f) * coverage_penalty
    end
    """ % (confidence, seed, lambda_cov)


def pinball_loss_julia(confidence):
    """Elementwise pinball (quantile) loss at level `confidence`, for fitting
    log|residual|: the minimizer is the conditional `confidence` quantile of
    log|residual|, i.e. the log of the |residual| quantile a conformal
    interval at that level needs (quantiles commute with the monotone log).
    """
    tau = f"{confidence:.4f}"
    return f"pinball(prediction, target) = max({tau} * (target - prediction), ({tau} - 1) * (target - prediction))"
