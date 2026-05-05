function [Tjet,info] = gmls_prolongate_psi_order3_mod(T,K,ell,J,opts)
%GMLS_PROLONGATE_PSI_ORDER3_HUANG Recursive SVD/GMLS jet prolongation.
%
% This follows the Huang paper workflow more closely than direct polynomial
% regression: estimate tangent frames, recover derivatives from those
% tangents, append the new jet coordinates, and repeat through order 3.
% By default the SVD/GMLS geometry is computed in nondimensional jet
% coordinates, then mapped back to physical coordinates before derivatives
% and normals are returned.

    if nargin < 2 || isempty(K), K = 50; end
    if nargin < 3 || isempty(ell), ell = 4; end
    if nargin < 4 || isempty(J), J = 2; end
    if nargin < 5, opts = struct(); end

    x = T.x(:);
    y = T.y(:);
    psi = T.psi(:);

    d = 2;
    Z0 = [x,y,psi];

    [frames0,stats0] = gmls_frames(Z0,d,K,ell,J,opts);
    [gPsi,cond0] = recover_coordinate_gradient(frames0,3,d);

    Z1 = [Z0,gPsi(:,1),gPsi(:,2)];

    [frames1,stats1] = gmls_frames(Z1,d,K,ell,J,opts);
    [gPx,cond1x] = recover_coordinate_gradient(frames1,4,d);
    [gPy,cond1y] = recover_coordinate_gradient(frames1,5,d);

    psi_xx = gPx(:,1);
    psi_xy = 0.5*(gPx(:,2) + gPy(:,1));
    psi_yy = gPy(:,2);

    Z2 = [Z1,psi_xx,psi_xy,psi_yy];

    [frames2,stats2] = gmls_frames(Z2,d,K,ell,J,opts);
    [gPxx,cond2xx] = recover_coordinate_gradient(frames2,6,d);
    [gPxy,cond2xy] = recover_coordinate_gradient(frames2,7,d);
    [gPyy,cond2yy] = recover_coordinate_gradient(frames2,8,d);

    psi_xxx = gPxx(:,1);
    psi_xxy = 0.5*(gPxx(:,2) + gPxy(:,1));
    psi_xyy = 0.5*(gPxy(:,2) + gPyy(:,1));
    psi_yyy = gPyy(:,2);

    Z3 = [Z2,psi_xxx,psi_xxy,psi_xyy,psi_yyy];

    [frames3,stats3] = gmls_frames(Z3,d,K,ell,J,opts);

    Tjet = T;
    Tjet.psi_x = Z3(:,4);
    Tjet.psi_y = Z3(:,5);
    Tjet.psi_xx = Z3(:,6);
    Tjet.psi_xy = Z3(:,7);
    Tjet.psi_yy = Z3(:,8);
    Tjet.psi_xxx = Z3(:,9);
    Tjet.psi_xxy = Z3(:,10);
    Tjet.psi_xyy = Z3(:,11);
    Tjet.psi_yyy = Z3(:,12);

    info = struct();
    info.Z0 = Z0;
    info.Z1 = Z1;
    info.Z2 = Z2;
    info.Z3 = Z3;
    info.tangents3 = frames3.T;
    info.normals3 = frames3.N;
    info.stats = {stats0,stats1,stats2,stats3};
    info.gradientConditionNumbers = {cond0,cond1x,cond1y,cond2xx,cond2xy,cond2yy};
end

function [frames,stats] = gmls_frames(Y,d,K,ell,J,opts)
    [nPts,ambientDim] = size(Y);
    K = min(K,nPts);
    if K <= d
        error('gmls_frames:StencilTooSmall','K must be larger than d.');
    end

    coordScale = frame_coordinate_scale(Y,opts);
    Ywork = bsxfun(@rdivide,Y,coordScale);

    idxAll = nearest_neighbors(Ywork,K);
    exps = monomial_exponents(d,ell);
    nMono = size(exps,1);

    Tstore = zeros(ambientDim,d,nPts);
    Nstore = zeros(ambientDim,ambientDim-d,nPts);
    rankV = zeros(nPts,1);
    correctionNorm = zeros(nPts,J);

    for i = 1:nPts
        idx = idxAll(i,:);
        M = (Ywork(idx,:) - Ywork(i,:))';

        [U,~,~] = svd(M,'econ');
        Tloc = U(:,1:d);
        Nloc = null(Tloc');

        for iter = 1:J
            if isempty(Nloc)
                break
            end

            Q = [Tloc,Nloc];
            proj = Q' * M(:,2:end);
            tau = proj(1:d,:)';
            normalDev = proj(d+1:end,:)';

            tauScale = max(abs(tau),[],1);
            tauScale(tauScale == 0) = 1;
            tauHat = bsxfun(@rdivide,tau,tauScale);

            V = vandermonde_from_exponents(tauHat,exps);
            rankV(i) = rank(V);
            B = V \ normalDev;

            Dpi = B(1:d,:)';
            Dpi = bsxfun(@rdivide,Dpi,tauScale);
            correctionNorm(i,iter) = norm(Dpi,'fro');

            Tnew = Tloc + Nloc*Dpi;
            [Tloc,~] = qr(Tnew,0);
            Nloc = null(Tloc');
        end

        Tphys = bsxfun(@times,Tloc,coordScale(:));
        Ncandidate = bsxfun(@rdivide,Nloc,coordScale(:));
        [Nphys,~] = qr(Ncandidate,0);

        Tstore(:,:,i) = Tphys;
        Nstore(:,:,i) = Nphys;
    end

    frames = struct('T',Tstore,'N',Nstore);
    stats = struct('rankV',rankV,'nMonomials',nMono,'K',K, ...
                   'ell',ell,'J',J,'coordinateScale',coordScale, ...
                   'correctionNorm',correctionNorm, ...
                   'maxCorrectionByIteration',max(correctionNorm,[],1));
end

function [grad,condA] = recover_coordinate_gradient(frames,col,d)
    nPts = size(frames.T,3);
    grad = zeros(nPts,d);
    condA = zeros(nPts,1);

    for i = 1:nPts
        Tloc = frames.T(:,:,i);
        A = Tloc(1:d,:)';
        b = Tloc(col,:)';
        condA(i) = cond(A);
        if rcond(A) < 1e-12
            grad(i,:) = (pinv(A)*b)';
        else
            grad(i,:) = (A\b)';
        end
    end
end

function idxAll = nearest_neighbors(Y,K)
    if exist('knnsearch','file') == 2
        idxAll = knnsearch(Y,Y,'K',K);
        return
    end

    nPts = size(Y,1);
    idxAll = zeros(nPts,K);
    for i = 1:nPts
        dist2 = sum(bsxfun(@minus,Y,Y(i,:)).^2,2);
        [~,idx] = sort(dist2,'ascend');
        idxAll(i,:) = idx(1:K);
    end
end

function scale = frame_coordinate_scale(Y,opts)
    useNondimensionalFrames = true;
    if isfield(opts,'UseNondimensionalFrames')
        useNondimensionalFrames = opts.UseNondimensionalFrames;
    end

    if ~useNondimensionalFrames
        scale = ones(1,size(Y,2));
        return
    end

    scale = max(Y,[],1) - min(Y,[],1);
        useRobustScale = false;
    if isfield(opts,'UseRobustScale')
        useRobustScale = opts.UseRobustScale;
    end
    if useRobustScale
        robustScale = local_iqr(Y);
        robustScale(robustScale == 0) = scale(robustScale == 0);
        scale = robustScale;
    end

    zeroCols = scale == 0;
    scale(zeroCols) = 1;
end

function s = local_iqr(Y)
    s = zeros(1,size(Y,2));
    for j = 1:size(Y,2)
        ys = sort(Y(:,j));
        n = numel(ys);
        q1 = ys(max(1,round(0.25*n)));
        q3 = ys(max(1,round(0.75*n)));
        s(j) = q3 - q1;
    end
end

function exps = monomial_exponents(d,ell)
    exps = eye(d);
    for total = 2:ell
        exps = [exps; fixed_degree_exponents(d,total)]; %#ok<AGROW>
    end
end

function exps = fixed_degree_exponents(d,total)
    if d == 1
        exps = total;
        return
    end

    exps = [];
    for k = total:-1:0
        tail = fixed_degree_exponents(d-1,total-k);
        exps = [exps; [k*ones(size(tail,1),1),tail]]; %#ok<AGROW>
    end
end

function V = vandermonde_from_exponents(tau,exps)
    n = size(tau,1);
    nMono = size(exps,1);
    V = ones(n,nMono);
    for j = 1:nMono
        for q = 1:size(tau,2)
            if exps(j,q) ~= 0
                V(:,j) = V(:,j).*tau(:,q).^exps(j,q);
            end
        end
    end
end
