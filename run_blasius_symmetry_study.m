%% Blasius data-driven Lie symmetry study
clear
clc
close all

%thisDir = fileparts(mfilename('fullpath'));
%addpath(thisDir)

rng(7)

% Keep this first test well-conditioned. After it passes, decrease nu and
% adjust yRange toward the physical thin-boundary-layer case.
nx = 50;
ny = 50;
Ue = 1.0;
nu = 1.5e-5;
xRange = [0.05 2.0];
yRange = [0.0 0.05];

K = 30; % Number of nearest neighbors for each local SVD/GMLS fit.
ell = 4;
J = 2;
runConvergence = true;
convergenceNN = [18 24 32];

cTrue = [0;2;0;0; 0;0;1;0; 0;0;0;1];
cTrue = cTrue / norm(cTrue);

eqOpts = struct('RowNormalize',true,'ColumnNormalize',true);
augOpts = struct('RowNormalize',true,'ColumnNormalize',true);

T = generate_blasius_streamfunction_data(nx,ny,Ue,nu,xRange,yRange,'uniform');
Texact = append_blasius_exact_jets(T);
interiorMask = interior_point_mask(T,0.15);
eqInteriorOpts = eqOpts;
eqInteriorOpts.PointMask = interiorMask;
normalOpts = struct('RowNormalize',true,'ColumnNormalize',true, ...
                    'PointMask',interiorMask);

fprintf('Interior points used for primary PDE/Huang metrics: %d of %d\n', ...
        nnz(interiorMask),height(T));

fprintf('\n--- Exact known-PDE control ---\n')
[~,pdeExactMeta] = build_Ptilde_from_known_blasius_pde(Texact,nu);
fprintf('||Ppde_exact*cTrue|| / ||Ppde_exact|| = %.3e\n', ...
        norm(pdeExactMeta.Praw*cTrue)/max(1,norm(pdeExactMeta.Praw,'fro')));

fprintf('\n--- Exact-derivative PDE-only nullspace diagnostic ---\n')
[PeqExact,eqExactMeta] = build_Ptilde_from_known_blasius_pde(Texact,nu,eqInteriorOpts);
eqExactResult = nullspace_report(PeqExact,eqExactMeta.colNorms,cTrue,1e-8);
fprintf('Exact PDE-only nullity = %d\n', eqExactResult.nullity);
fprintf('Exact PDE-only raw-vector alignment = %.6f\n', eqExactResult.rawAlignment);
fprintf('Exact PDE-only sin(theta) = %.3e\n', eqExactResult.angleError);
disp('Exact PDE-only vector scaled so y coefficient is 1:')
disp(scale_by_y(eqExactResult.cRaw).')

fprintf('\n--- Exact-derivative augmented known-PDE/wall diagnostic ---\n')
[PaugExact,augExactMeta] = build_Ptilde_blasius_boundary_control(Texact,nu,augOpts);
augExactResult = nullspace_report(PaugExact,augExactMeta.colNorms,cTrue,1e-8);
fprintf('Exact augmented nullity = %d\n', augExactResult.nullity);
fprintf('Exact augmented raw-vector alignment = %.6f\n', augExactResult.rawAlignment);
disp('Exact augmented best-aligned raw vector:')
disp(augExactResult.cRaw.')
disp('Exact augmented vector scaled so y coefficient is 1:')
disp(scale_by_y(augExactResult.cRaw).')

gmlsOpts = struct();
gmlsOpts.UseNondimensionalFrames = true;
gmlsOpts.UseRobustScale = false;

fprintf('\n--- Numerical SVD/GMLS prolongation ---\n')
[Tjet,info] = gmls_prolongate_psi_order3_mod(T,K,ell,J,gmlsOpts);
print_jet_errors(Tjet,Texact)

fprintf('\nMax gradient-system condition numbers by stage:\n')
for q = 1:numel(info.gradientConditionNumbers)
    fprintf('  stage %d: %.3e\n', q, max(info.gradientConditionNumbers{q}));
end

fprintf('\nMax GMLS tangent correction by jet level and iteration:\n')
for q = 1:numel(info.stats)
    fprintf('  jet level %d:', q-1);
    fprintf(' %.3e', info.stats{q}.maxCorrectionByIteration);
    fprintf('\n');
end

fprintf('\nCoordinate scales used for nondimensional SVD/GMLS frames:\n')
for q = 1:numel(info.stats)
    fprintf('  jet level %d:', q-1);
    fprintf(' %.3e', info.stats{q}.coordinateScale);
    fprintf('\n');
end

fprintf('\n--- Known-PDE check using estimated jets ---\n')
[~,pdeEstMeta] = build_Ptilde_from_known_blasius_pde(Tjet,nu);
fprintf('||Ppde_est*cTrue|| / ||Ppde_est|| = %.3e\n', ...
        norm(pdeEstMeta.Praw*cTrue)/max(1,norm(pdeEstMeta.Praw,'fro')));

fprintf('\n--- GMLS-derivative PDE-only nullspace diagnostic ---\n')
[PeqGmls,eqGmlsMeta] = build_Ptilde_from_known_blasius_pde(Tjet,nu,eqInteriorOpts);
eqGmlsResult = nullspace_report(PeqGmls,eqGmlsMeta.colNorms,cTrue,1e-8);
fprintf('GMLS PDE-only nullity = %d\n', eqGmlsResult.nullity);
fprintf('GMLS PDE-only raw-vector alignment = %.6f\n', eqGmlsResult.rawAlignment);
fprintf('GMLS PDE-only sin(theta) = %.3e\n', eqGmlsResult.angleError);
disp('GMLS PDE-only best-aligned raw vector:')
disp(eqGmlsResult.cRaw.')


disp('GMLS PDE-only vector scaled so y coefficient is 1:')
disp(scale_by_y(eqGmlsResult.cRaw).')
%-----------------------------------------------------------------
%-----------------------------------------------------------------
fprintf('\n--- GMLS PDE + freestream diagnostic ---\n')

ffOpts = eqInteriorOpts;
ffOpts.FarfieldFraction = 0.05;

[PgmlsFree,freeMeta] = build_Ptilde_blasius_freestream_control(Tjet,nu,ffOpts);
freeResult = nullspace_report(PgmlsFree,freeMeta.colNorms,cTrue,1e-8);

fprintf('GMLS PDE+freestream nullity = %d\n', freeResult.nullity);
fprintf('GMLS PDE+freestream alignment = %.6f\n', freeResult.rawAlignment);
fprintf('GMLS PDE+freestream sin(theta) = %.3e\n', freeResult.angleError);

disp('GMLS PDE+freestream best-aligned raw vector:')
disp(freeResult.cRaw.')

disp('GMLS PDE+freestream vector scaled so y coefficient is 1:')
disp(scale_by_y(freeResult.cRaw).')
%--------------------------------------------------------------------------
fprintf('\n--- GMLS PDE + freestream scaling-subspace diagnostic ---\n')

scaleCols = [2 7 12];

[PgmlsFree,freeMeta] = build_Ptilde_blasius_freestream_control(Tjet,nu,ffOpts);

Pscale = PgmlsFree(:,scaleCols);
[~,~,Vscale] = svd(Pscale,'econ');

aScaled = Vscale(:,end);

% Convert normalized coefficients back to raw coefficients.
aRaw = aScaled ./ freeMeta.colNorms(scaleCols).';

% Scale so c7 = 1.
aRaw = aRaw / aRaw(2);

cScaleOnly = zeros(12,1);
cScaleOnly(scaleCols) = aRaw;
cScaleOnly(abs(cScaleOnly) < 1e-8) = 0;

disp('GMLS PDE+freestream scaling-only raw vector:')
disp(cScaleOnly.')
%-------------------------------------------------------------------------------------
fprintf('\n--- Exact PDE + freestream scaling-subspace diagnostic ---\n')

scaleCols = [2 7 12];

[PexactFree,exactFreeMeta] = build_Ptilde_blasius_freestream_control(Texact,nu,ffOpts);

PscaleExact = PexactFree(:,scaleCols);
[~,~,VscaleExact] = svd(PscaleExact,'econ');

aScaled = VscaleExact(:,end);

% Convert from column-normalized coordinates back to raw coefficients.
aRaw = aScaled ./ exactFreeMeta.colNorms(scaleCols).';

% Scale so c7 = 1.
aRaw = aRaw / aRaw(2);

cScaleExact = zeros(12,1);
cScaleExact(scaleCols) = aRaw;
cScaleExact(abs(cScaleExact) < 1e-8) = 0;

disp('Exact PDE+freestream scaling-only raw vector:')
disp(cScaleExact.')
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

fprintf('\n--- Data-driven jet-normal invariance matrix check ---\n')
[Pnormal,normalMeta] = build_Ptilde_from_jet_normals_blasius( ...
    Tjet,info.normals3,normalOpts);

normalResult = nullspace_report(Pnormal,normalMeta.colNorms,cTrue,1e-8);
fprintf('Smallest relative singular value = %.3e\n', normalResult.relSingularValues(end));
fprintf('Detected nullity = %d\n', normalResult.nullity);
fprintf('Best raw-vector alignment with Blasius scaling = %.6f\n', normalResult.rawAlignment);
fprintf('Jet-normal invariance sin(theta) = %.3e\n', normalResult.angleError);
fprintf('Projection of true scaling into normalized nullspace = %.6f\n', normalResult.subspaceAlignment);
disp('Normal-space raw vector:')
disp(normalResult.cRaw.')

disp('Normal-space vector scaled by dominant coefficient:')
disp(scale_by_dominant(normalResult.cRaw).')

%--------------------------------------------------------------------------
fprintf('\n--- Data-driven jet-normal scaling-subspace diagnostic ---\n')

scaleCols = [2 7 12];

PscaleNormal = Pnormal(:,scaleCols);

[~,~,VscaleNormal] = svd(PscaleNormal,'econ');

aScaled = VscaleNormal(:,end);

% Convert from column-normalized coordinates back to raw coefficients.
aRaw = aScaled ./ normalMeta.colNorms(scaleCols).';

% Scale so c7 = 1. Since scaleCols = [2 7 12], c7 is entry 2.
aRaw = aRaw / aRaw(2);

cNormalScaleOnly = zeros(12,1);
cNormalScaleOnly(scaleCols) = aRaw;
cNormalScaleOnly(abs(cNormalScaleOnly) < 1e-8) = 0;

disp('Data-driven jet-normal scaling-only raw vector:')
disp(cNormalScaleOnly.')

targetScale = [2;1;1];
targetScale = targetScale / norm(targetScale);

aNorm = aRaw / norm(aRaw);

fprintf('Scaling-subspace alignment = %.6f\n', abs(dot(aNorm,targetScale)));
fprintf('Scaling-subspace residual = %.3e\n', ...
        norm(PscaleNormal*aScaled)/max(1,norm(PscaleNormal,'fro')));
%------------------------------------------------------------------------------

fprintf('\n--- Optional augmented known-PDE/wall diagnostic ---\n')
[Paug,augMeta] = build_Ptilde_blasius_boundary_control(Tjet,nu,augOpts);
augResult = nullspace_report(Paug,augMeta.colNorms,cTrue,1e-8);
fprintf('Augmented nullity = %d\n', augResult.nullity);
fprintf('Augmented raw-vector alignment = %.6f\n', augResult.rawAlignment);
disp('Augmented best-aligned raw vector:')
disp(augResult.cRaw.')
disp('Augmented vector scaled so y coefficient is 1:')
disp(scale_by_y(augResult.cRaw).')


figure(1)
semilogy(normalResult.relSingularValues,'o-','LineWidth',1.5)
grid on
xlabel('Index')
ylabel('Relative singular value')
title('Data-driven jet-normal Blasius invariance matrix')

figure(2)
scatter(Tjet.x,Tjet.y,24,Tjet.psi_y,'filled')
axis tight
grid on
colorbar
xlabel('x')
ylabel('y')
title('Estimated \psi_y from recursive SVD/GMLS')

if runConvergence
    conv = run_convergence_check(convergenceNN,Ue,nu,xRange,yRange, ...
                                 K,ell,J,gmlsOpts,eqOpts,cTrue);

    figure(3)
    loglog(conv.numSamples,conv.errExactPde,'o-','LineWidth',1.8)
    hold on
    loglog(conv.numSamples,conv.errGmlsPde,'s-','LineWidth',1.8)
    hold on
    loglog(conv.numSamples,conv.errJetNormal,'^-','LineWidth',1.8)
    grid on
    xlabel('Number of samples')
    ylabel('sin(\theta)')
    legend('Exact PDE-only','GMLS PDE-only','Data-driven jet-normal', ...
           'Location','best')
    title('Blasius generator recovery: interior points only')
end
%---------------------------------------------------------------------------
%% Presentation / interpretation report

genLabels = {};
genVectors = [];

genLabels{end+1} = 'Analytic-jet PDE control';
genVectors(:,end+1) = eqExactResult.cRaw;

genLabels{end+1} = 'GMLS-jet PDE control';
genVectors(:,end+1) = eqGmlsResult.cRaw;

if exist('cNormalScaleOnly','var')
    genLabels{end+1} = 'Jet-normal scaling subspace';
    genVectors(:,end+1) = cNormalScaleOnly;
else
    genLabels{end+1} = 'Data-driven jet-normal';
    genVectors(:,end+1) = normalResult.cRaw;
end

if exist('cScaleExact','var')
    genLabels{end+1} = 'Analytic-jet freestream control';
    genVectors(:,end+1) = cScaleExact;
end

if exist('cScaleOnly','var')
    genLabels{end+1} = 'GMLS-jet freestream control';
    genVectors(:,end+1) = cScaleOnly;
end

fprintf('\n--- Presentation generator summary ---\n')

scaledVectors = genVectors;
for j = 1:size(scaledVectors,2)
    if abs(scaledVectors(7,j)) > 1e-14
        scaledVectors(:,j) = scaledVectors(:,j) / scaledVectors(7,j);
    end
    scaledVectors(abs(scaledVectors(:,j)) < 1e-10,j) = 0;
end

c2 = scaledVectors(2,:).';
c7 = scaledVectors(7,:).';
c12 = scaledVectors(12,:).';

pdeBalance = c2 - c7 - c12;
freestreamBalance = c12 - c7;

summaryTable = table(genLabels(:),c2,c7,c12,pdeBalance,freestreamBalance, ...
    'VariableNames',{'Generator','c2_xScale','c7_yScale', ...
                     'c12_psiScale','PDE_c2_minus_c7_minus_c12', ...
                     'Freestream_c12_minus_c7'});

disp(summaryTable)

fprintf('\n--- Jet derivative error summary ---\n')

jetNames = {'psi_x','psi_y','psi_xx','psi_xy','psi_yy', ...
            'psi_xxx','psi_xxy','psi_xyy','psi_yyy'};

relRms = zeros(numel(jetNames),1);
maxAbs = zeros(numel(jetNames),1);
relMax = zeros(numel(jetNames),1);

for i = 1:numel(jetNames)
    a = Tjet.(jetNames{i});
    b = Texact.(jetNames{i});
    e = a - b;

    relRms(i) = sqrt(mean(e.^2)) / max(1e-14,sqrt(mean(b.^2)));
    maxAbs(i) = max(abs(e));
    relMax(i) = maxAbs(i) / max(1e-14,max(abs(b)));
end

jetErrorTable = table(jetNames(:),relRms,maxAbs,relMax, ...
    'VariableNames',{'Derivative','RelativeRMS','MaxAbsError','RelativeMax'});

disp(jetErrorTable)

%% Plot 4: similarity profile and streamfunction

figure(4)
clf
tiledlayout(1,2,'TileSpacing','compact','Padding','compact')

nexttile
scatter(T.eta,T.u./T.Ue,18,[0.2 0.2 0.2],'filled')
hold on
scatter(Tjet.eta,Tjet.psi_y./Tjet.Ue,12,[0.0 0.45 0.74],'filled')
scatter(Texact.eta,Texact.psi_y./Texact.Ue,8,[0.85 0.33 0.10],'filled')
grid on
xlabel('\eta')
ylabel('u/U_e = \psi_y/U_e')
title('Similarity profile')
legend('Generated u/U_e','GMLS \psi_y/U_e','Exact \psi_y/U_e','Location','southeast')

nexttile
scatter(T.x,T.y,20,T.psi,'filled')
axis tight
grid on
colorbar
xlabel('x')
ylabel('y')
title('Streamfunction field \psi')

%% Plot 5: derivative error maps

figure(5)
clf
plotNames = {'psi_y','psi_xy','psi_yyy'};
plotTitles = {'\psi_y relative error','\psi_{xy} relative error','\psi_{yyy} relative error'};

tiledlayout(1,3,'TileSpacing','compact','Padding','compact')

for i = 1:numel(plotNames)
    a = Tjet.(plotNames{i});
    b = Texact.(plotNames{i});
    e = abs(a-b) / max(1e-14,max(abs(b)));

    nexttile
    scatter(Tjet.x,Tjet.y,20,e,'filled')
    axis tight
    grid on
    colorbar
    xlabel('x')
    ylabel('y')
    title(plotTitles{i})
end

%% Plot 6: recovered generator coefficients

figure(6)
clf
bar([c2 c7 c12])
grid on
set(gca,'XTick',1:numel(genLabels),'XTickLabel',genLabels)
xtickangle(25)
ylabel('Coefficient value after c7 normalization')
title('Recovered scaling generator coefficients')
legend('c2: x scale','c7: y scale','c12: \psi scale','Location','best')
yline(2,'--','target c2')
yline(1,':','target c7,c12')

%% Plot 7: physics balance residuals

figure(7)
clf
bar([pdeBalance freestreamBalance])
grid on
set(gca,'XTick',1:numel(genLabels),'XTickLabel',genLabels)
xtickangle(25)
ylabel('Balance residual')
title('Physics checks for scaling generator')
legend('PDE: c2 - c7 - c12','Freestream: c12 - c7','Location','best')
yline(0,'k-')

%% Plot 8: nullspace diagnostic spectra

figure(8)
clf
hold on

if exist('eqExactResult','var')
    semilogy(eqExactResult.relSingularValues,'o-','LineWidth',1.4)
end
if exist('eqGmlsResult','var')
    semilogy(eqGmlsResult.relSingularValues,'s-','LineWidth',1.4)
end
if exist('normalResult','var')
    semilogy(normalResult.relSingularValues,'^-','LineWidth',1.4)
end

grid on
xlabel('Index')
ylabel('Relative singular value')
title('Nullspace diagnostic spectra')
legend('Analytic-jet PDE control', ...
       'GMLS-jet PDE control', ...
       'Full jet-normal matrix', ...
       'Location','best')

%% Plot 9: jet-normal scaling-subspace alignment convergence

if exist('conv','var')
    figure(9)
    clf

    alignExactPde = sqrt(max(0,1 - conv.errExactPde.^2));
    alignGmlsPde = sqrt(max(0,1 - conv.errGmlsPde.^2));

    plot(conv.numSamples,alignExactPde,'o-','LineWidth',1.8)
    hold on
    plot(conv.numSamples,alignGmlsPde,'s-','LineWidth',1.8)
    plot(conv.numSamples,conv.alignJetNormal,'^-','LineWidth',1.8)

    grid on
    xlabel('Number of samples')
    ylabel('Alignment with target generator')
    ylim([0 1.05])
    title('Generator recovery alignment convergence')
    legend('Analytic-jet PDE control', ...
        'GMLS-jet PDE control', ...
        'Jet-normal scaling subspace', ...
        'Location','southeast')

    yline(1,'--','target alignment','LabelHorizontalAlignment','left')
end

%% Plot 10: data prolongation / jet-space visualization

figure(10)
clf
tiledlayout(1,3,'TileSpacing','compact','Padding','compact')

% Original data manifold: [x,y,psi]
nexttile
scatter3(T.x,T.y,T.psi,18,T.eta,'filled')
grid on
xlabel('x')
ylabel('y')
zlabel('\psi')
title('Original data: (x,y,\psi)')
cb = colorbar;
cb.Label.String = '\eta';
view(35,25)

% First jet projection: [x,y,psi_y]
nexttile
scatter3(Tjet.x,Tjet.y,Tjet.psi_y,18,Tjet.eta,'filled')
grid on
xlabel('x')
ylabel('y')
zlabel('\psi_y')
title('First prolongation: (x,y,\psi_y)')
cb = colorbar;
cb.Label.String = '\eta';
view(35,25)
set(gcf,'Toolbar','none')

% PDE-relevant third-jet projection
nexttile
scatter3(Tjet.psi_xy,Tjet.psi_yy,Tjet.psi_yyy,18,Tjet.eta,'filled')
grid on
xlabel('\psi_{xy}')
ylabel('\psi_{yy}')
zlabel('\psi_{yyy}')
title('PDE-relevant jet projection')
cb = colorbar;
cb.Label.String = '\eta';
view(35,25)
set(gcf,'Toolbar','none')

%% Plot 11: exact vs GMLS jet-space projection

figure(11)
clf
tiledlayout(1,2,'TileSpacing','compact','Padding','compact')

nexttile
scatter3(Texact.psi_xy,Texact.psi_yy,Texact.psi_yyy,18,Texact.eta,'filled')
grid on
xlabel('\psi_{xy}')
ylabel('\psi_{yy}')
zlabel('\psi_{yyy}')
title('Analytic jet projection')
cb = colorbar;
cb.Label.String = '\eta';
view(35,25)

nexttile
scatter3(Tjet.psi_xy,Tjet.psi_yy,Tjet.psi_yyy,18,Tjet.eta,'filled')
grid on
xlabel('\psi_{xy}')
ylabel('\psi_{yy}')
zlabel('\psi_{yyy}')
title('GMLS-prolonged jet projection')
cb = colorbar;
cb.Label.String = '\eta';
view(35,25)

%% Plot 12: exact vs GMLS-estimated streamwise velocity u

figure(12)
clf
tiledlayout(1,2,'TileSpacing','compact','Padding','compact')

uMin = min([Texact.psi_y; Tjet.psi_y]);
uMax = max([Texact.psi_y; Tjet.psi_y]);

nexttile
scatter(Texact.x,Texact.y,24,Texact.psi_y,'filled')
axis tight
grid on
colorbar
clim([uMin uMax])
xlabel('x')
ylabel('y')
title('Exact streamwise velocity')
subtitle('u = \psi_y')

nexttile
scatter(Tjet.x,Tjet.y,24,Tjet.psi_y,'filled')
axis tight
grid on
colorbar
clim([uMin uMax])
xlabel('x')
ylabel('y')
title('GMLS-estimated streamwise velocity')
subtitle('u = \psi_y')

%---------------------------------------------------------------------------
function print_jet_errors(Tjet,Texact)
    names = {'psi_x','psi_y','psi_xx','psi_xy','psi_yy', ...
             'psi_xxx','psi_xxy','psi_xyy','psi_yyy'};
    for i = 1:numel(names)
        a = Tjet.(names{i});
        b = Texact.(names{i});
        rel = sqrt(mean((a-b).^2)) / max(1e-14,sqrt(mean(b.^2)));
        fprintf('  %-8s relative RMS error: %.3e\n', names{i}, rel);
    end
end

function result = nullspace_report(P,colNorms,cTrue,tol)
    [~,S,V] = svd(P,'econ');
    s = diag(S);
    rel = s / max(s(1),eps);

    nullIdx = find(rel < tol);
    if isempty(nullIdx)
        nullIdx = numel(s);
    end

    colNorms = colNorms(:);
    cTrueScaled = colNorms .* cTrue(:);
    cTrueScaled = cTrueScaled / norm(cTrueScaled);

    Nscaled = V(:,nullIdx);
    projScaled = Nscaled * (Nscaled' * cTrueScaled);
    subspaceAlignment = norm(projScaled);

    if norm(projScaled) < 1e-14
        cScaled = V(:,end);
    else
        cScaled = projScaled / norm(projScaled);
    end

    cRaw = cScaled ./ colNorms;
    cRaw = cRaw / norm(cRaw);

    result = struct();
    result.relSingularValues = rel;
    result.nullity = numel(nullIdx);
    result.cRaw = cRaw;
    result.rawAlignment = abs(dot(cRaw,cTrue(:)));
    result.subspaceAlignment = subspaceAlignment;
    result.angleError = sqrt(max(0,1 - result.rawAlignment^2));
end

function cScaled = scale_by_y(c)
    cScaled = c(:);
    if abs(cScaled(7)) > 1e-14
        cScaled = cScaled / cScaled(7);
    end
    cScaled(abs(cScaled) < 1e-8) = 0;
end

function mask = interior_point_mask(T,frac)
    x = T.x(:);
    y = T.y(:);
    xr = max(x) - min(x);
    yr = max(y) - min(y);

    mask = x > min(x) + frac*xr & x < max(x) - frac*xr & ...
           y > min(y) + frac*yr & y < max(y) - frac*yr;

    if nnz(mask) < max(20,0.25*numel(mask))
        mask = y > min(y) + max(1e-12,1e-8*max(1,yr));
    end
end

function conv = run_convergence_check(NN,Ue,nu,xRange,yRange,K,ell,J, ...
                                      gmlsOpts,eqOpts,cTrue)
    errExactPde = zeros(size(NN));
    errGmlsPde = zeros(size(NN));
    errJetNormal = zeros(size(NN));
    alignJetNormal = zeros(size(NN));
    numSamples = zeros(size(NN));

    fprintf('\n--- Interior convergence check ---\n')
    for q = 1:numel(NN)
        fprintf('Convergence case %d/%d: N = %d\n', q,numel(NN),NN(q));

        Tq = generate_blasius_streamfunction_data(NN(q),NN(q),Ue,nu,xRange,yRange,'uniform');
        TexactQ = append_blasius_exact_jets(Tq);
        maskQ = interior_point_mask(Tq,0.05);
        numSamples(q) = height(Tq);

        eqOptsQ = eqOpts;
        eqOptsQ.PointMask = maskQ;

        normalOptsQ = struct('RowNormalize',true,'ColumnNormalize',true, ...
                             'PointMask',maskQ);

        Kq = min(K,max(25,round(0.05*height(Tq))));

        [PeqExactQ,eqExactMetaQ] = build_Ptilde_from_known_blasius_pde(TexactQ,nu,eqOptsQ);
        exactQ = nullspace_report(PeqExactQ,eqExactMetaQ.colNorms,cTrue,1e-8);
        errExactPde(q) = exactQ.angleError;

        [TjetQ,infoQ] = gmls_prolongate_psi_order3_mod(Tq,Kq,ell,J,gmlsOpts);

        [PeqGmlsQ,eqGmlsMetaQ] = build_Ptilde_from_known_blasius_pde(TjetQ,nu,eqOptsQ);
        gmlsQ = nullspace_report(PeqGmlsQ,eqGmlsMetaQ.colNorms,cTrue,1e-8);
        errGmlsPde(q) = gmlsQ.angleError;

        [PnormalQ,normalMetaQ] = build_Ptilde_from_jet_normals_blasius( ...
            TjetQ,infoQ.normals3,normalOptsQ);

        % Jet-normal scaling-subspace diagnostic
        scaleCols = [2 7 12];

        PscaleNormalQ = PnormalQ(:,scaleCols);
        [~,~,VscaleNormalQ] = svd(PscaleNormalQ,'econ');

        aScaledQ = VscaleNormalQ(:,end);

        % Convert from column-normalized coordinates back to raw coefficients
        aRawQ = aScaledQ ./ normalMetaQ.colNorms(scaleCols).';

        % Scale so c7 = 1
        aRawQ = aRawQ / aRawQ(2);

        cJetNormalScaleQ = zeros(12,1);
        cJetNormalScaleQ(scaleCols) = aRawQ;
        cJetNormalScaleQ = cJetNormalScaleQ / norm(cJetNormalScaleQ);

        alignJetNormal(q) = abs(dot(cJetNormalScaleQ,cTrue));
        errJetNormal(q) = sqrt(max(0,1 - alignJetNormal(q)^2));

        fprintf('  sin(theta): exact PDE %.3e, GMLS PDE %.3e, jet-normal %.3e\n', ...
                errExactPde(q),errGmlsPde(q),errJetNormal(q));
    end

    conv = struct();
    conv.NN = NN;
    conv.numSamples = numSamples;
    conv.errExactPde = errExactPde;
    conv.errGmlsPde = errGmlsPde;
    conv.errJetNormal = errJetNormal;
    conv.alignJetNormal = alignJetNormal;
end

function cScaled = scale_by_dominant(c)
    cScaled = c(:);
    [~,idx] = max(abs(cScaled));
    if abs(cScaled(idx)) > 1e-14
        cScaled = cScaled / cScaled(idx);
    end
    cScaled(abs(cScaled) < 1e-8) = 0;
end
