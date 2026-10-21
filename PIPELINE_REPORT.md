# Báo cáo kiến trúc và luồng hoạt động của Handwriting GAN pipeline

## 1. Kết luận nhanh

Repository này là một biến thể của **VATr (Visual Archetypes-based Transformer)** để sinh ảnh chữ viết tay ở mức từ (word-level) theo phong cách của một người viết.

Luồng chính thực tế trong mã là:

```text
files/<DATASET>-32.pickle
        |
        v
TextDataset / TextDatasetval
  - 15 ảnh mẫu cùng writer -> simg [B, 15, 32, 192]
  - 1 ảnh thật + nhãn       -> img, label, writer id
        |
        +------------------------------+
        |                              |
        v                              v
ResNet-18 style encoder          text -> character IDs
        |                              |
Transformer encoder              Unifont 16x16 glyphs -> Linear
        |                              |
        +-------- Transformer decoder -+
                          |
                     FCN decoder
                          |
              ảnh giả [B, 1, 32, 16 * số ký tự]
                          |
          +---------------+----------------+
          |               |                |
     GAN critic       OCR/CTC critic   Writer classifier
       netD              netOCR             netW
```

Ba entry point chính:

- `train.py`: huấn luyện VATr bằng bốn pha cập nhật xen kẽ.
- `generator.py`: API/CLI sinh ảnh từ một thư mục ảnh phong cách và một hoặc nhiều dòng text.
- `generate_fakes.py`: xuất cặp ảnh thật/giả theo writer để đánh giá FID hoặc đánh giá ngoài.

**Trạng thái hiện tại:** mã parse cú pháp thành công, nhưng repository **chưa tự chạy end-to-end** vì thiếu dataset/checkpoint được mã tham chiếu, README mô tả một pipeline dữ liệu khác với code, và có một số dependency/lỗi runtime được liệt kê ở mục 11.

## 2. Cấu trúc thư mục và ý nghĩa file

```text
pipeline/
|-- train.py                    entry point huấn luyện
|-- generator.py                API + CLI inference cho người dùng
|-- generate_fakes.py           sinh tập Real/Fake để đánh giá
|-- mytext.txt                  đoạn text cố định dùng preview mỗi epoch
|-- requirements*.txt           dependency chính/tùy chọn
|-- README.md                   hướng dẫn dự kiến, hiện lệch với implementation
|-- Groundtruth/                split/nhãn IAM dạng writer,image-id text
|-- corpora_english/            các word list phục vụ đánh giá in-vocab/OOV
|-- data/                       loader pickle và script chuẩn bị dữ liệu cũ
|-- dataset/                    hai placeholder rỗng, không được dùng
|-- models/                     generator, discriminator, OCR và layer hỗ trợ
|-- util/                       loss, logging, distributed, HTML/Visdom
`-- files/                      từ điển, Unifont, sơ đồ, ảnh mẫu và output mẫu
```

### File dữ liệu và tài sản

| Đường dẫn | Ý nghĩa |
|---|---|
| `Groundtruth/gan.iam.tr_va.gt.filter27` | Ground truth train+validation IAM, mỗi dòng dạng `writer_id,image_id text`. |
| `Groundtruth/gan.iam.test.gt.filter27` | Ground truth test IAM cùng định dạng. |
| `corpora_english/brown-azAZ.tr` | 22.553 từ từ Brown corpus, dùng làm corpus in-vocabulary/training trong các quy trình đánh giá ngoài. |
| `corpora_english/in_vocab.subset.tro.37` | 114 từ in-vocabulary chọn lọc. |
| `corpora_english/oov.common_words` | 79 từ OOV phổ biến. |
| `corpora_english/oov_words.txt` | 400 từ OOV. |
| `files/english_words.txt` | Lexicon rất lớn, 466.550 dòng. `VATr` lọc từ dài dưới 20 ký tự và chỉ giữ ký tự thuộc alphabet rồi lấy ngẫu nhiên để huấn luyện generator. |
| `files/unifont.pickle` | 55.659 glyph Unicode; mỗi phần tử có `idx` và ma trận bitmap Boolean 16x16. Đây là nguồn biểu diễn ký tự cho `UnifontModule`. |
| `files/style_samples/00`, `01`, `02` | Ba writer/style mẫu, lần lượt 22, 21, 21 ảnh; dùng trực tiếp bởi `generator.py`. |
| `files/output_00.png` ... `02.png` | Output inference mẫu, ảnh grayscale cao 32 px. |
| `files/model.pdf`, `model_dark.png`, `model_light.png` | Sơ đồ kiến trúc ở dạng PDF và PNG cho nền tối/sáng; không được code import. |
| `mytext.txt` | Đoạn văn preview. Constructor `VATr` encode đoạn này và `_generate_page()` sinh trang so sánh mỗi epoch. |
| `dataset/dataset.py`, `dataset/dataloader.py` | File 0 byte; placeholder/di tích, không tham gia pipeline. |
| `.gitignore` | Bỏ qua cache, virtual environment, checkpoint/output và dữ liệu raw/processed. |
| `LICENSE` | Giấy phép MIT. |

Các file quan trọng nhưng **không có trong repository**: `files/IAM-32.pickle`, `files/CVL-32.pickle`, `files/vatr.pth`, `files/resnet_18_pretrained.pth`, và `scripts/prepare_iam.py`.

## 3. Hợp đồng dữ liệu

### Cấu trúc pickle mà loader thực sự cần

```python
{
    "train": {
        writer_id: [
            {"img": PIL.Image, "label": str, "img_id": int},
            ...
        ],
        ...
    },
    "test": { ... cùng cấu trúc ... }
}
```

`img_id` chỉ bắt buộc với `FidDataset`; `TextDataset` huấn luyện chỉ dùng `img` và `label`.

### Một batch sau `TextCollator`

| Key | Kiểu/shape | Vai trò |
|---|---|---|
| `simg` | float tensor `[B,N,32,192]`, thường `N=15` | N ảnh style cùng writer, chuẩn hóa về `[-1,1]`. |
| `swids` | float tensor `[B,N]` | Chiều rộng gốc của từng ảnh style. Chỉ dùng khi render trang preview. |
| `img` | float tensor `[B,1,32,Wmax]` | Ảnh word thật, pad trắng theo batch. |
| `label` | list các `bytes` | Nhãn word thật. |
| `wcl` | float tensor `[B]` | Chỉ số writer trong dataset; `netW` ép sang `long`. |
| `idx` | list | ID ảnh; placeholder khi train, số thật khi xuất FID. |
| `img_path` | list | Hiện chỉ là chuỗi placeholder `img_path`. |

Ảnh được đổi grayscale, `ToTensor`, rồi normalize với mean/std 0.5. Nền trắng thành gần `+1`, nét đen thành gần `-1`.

### Shape trong generator

1. `simg [B,N,32,192]` được reshape thành `[B*N,1,32,192]`.
2. ResNet-18 bỏ average pool và FC, tạo feature `[B*N,512,1,6]`.
3. Ghép lại thành chuỗi style token xấp xỉ `[N*6,B,512]` và đưa qua Transformer encoder.
4. Text được `strLabelConverter` biến thành ID `[B,L]` (0 là padding/CTC blank).
5. `UnifontModule` lấy bitmap 16x16 = 256 chiều cho mỗi ID, rồi Linear thành 512 chiều.
6. Transformer decoder cross-attend query ký tự với memory phong cách.
7. Linear 512 -> 4096; reshape thành feature map `[B,512,4,2L]`.
8. Ba lần upsample x2 trong `FCNDecoder` tạo ảnh `[B,1,32,16L]`, `tanh` trong `[-1,1]`.

## 4. Pipeline chuẩn bị dữ liệu

### `data/prepare_data.py`

Đây là script chuẩn bị dữ liệu **phiên bản cũ**, hỗ trợ IAM/CVL/RIMES, vừa có khả năng ghi LMDB vừa gom `PIL.Image` vào dictionary. Main block chứa đường dẫn Linux/NFS hard-code nên không chạy nguyên trạng trên workspace hiện tại.

- `checkImageIsValid(imageBin)`: decode bytes bằng OpenCV grayscale và xác nhận ảnh có diện tích khác 0.
- `writeCache(env, cache)`: ghi một dictionary key/value vào một transaction LMDB; tự encode `str` thành bytes.
- `find_rot_angle(idx_letters)`: dùng PCA trên tọa độ pixel/chữ và suy ra góc của principal component đầu tiên.
- `read_data_from_folder(folder_path)`: liệt kê file, lấy phần tên trước `_` làm label.
- `read_data_from_file(file_path)`: đọc file hai dòng một cặp: dòng path rồi dòng label.
- `show_demo(...)`: in một số cặp ảnh/label để người vận hành kiểm tra.
- `create_img_label_list(...)`: tìm ảnh và label theo layout riêng của CVL, IAM hoặc RIMES; với IAM đọc partition list + XML, có thể chọn một author và loại punctuation; trả về path list, label list, output dir, author ID.
- `createDataset(...)`: resize ảnh theo `charResize`, `keepRatio` hoặc giữ nguyên; thêm margin; gom ảnh theo author vào `IMG_DATA`; encode TIFF và ghi LMDB với key `image-%09d`, `label-%09d`, `num-samples`. Hàm xóa toàn bộ output LMDB cũ trước khi tạo lại.
- `createDict(...)`: sinh lexicon unique (`stratified`) và đầy đủ có lặp (`NOTstratified`).
- `printAlphabet(label_list)`: in các ký tự duy nhất theo thứ tự xuất hiện.
- Main block: đọc hai Groundtruth IAM, ánh xạ word image/XML sang writer, sau đó loop author với các path hard-code. Đoạn cuối không thực sự `pickle.dump`, nên chưa tạo được `IAM-32.pickle` mà training yêu cầu.

### `data/create_data.py`

Không phải một pipeline hoàn chỉnh. File chỉ import nhiều thư viện và thực hiện một `glob.glob()` trên đường dẫn CVL NFS hard-code; kết quả không được lưu. Có thể coi là scratch/placeholder.

## 5. Dataset và collator

### `data/dataset.py`

- `get_transform(grayscale=False, convert=True)`: dựng torchvision transform; tùy chọn grayscale, tensor hóa và normalize `[-1,1]`.

#### `TextDataset`

- `__init__`: đọc nhánh `train` của pickle, bỏ writer key `'None'`, tính alphabet và danh sách writer, gắn `TextCollator`.
- `__len__`: trả `max(số writer, min_virtual_size)` để có thể kéo dài epoch ảo.
- `num_writers`: số writer thật.
- `__getitem__`: map index về writer; lấy có hoàn lại N ảnh style và một ảnh thật ngẫu nhiên; pad/crop mỗi style về width 192; trả dictionary batch item.

#### `TextDatasetval`

Ba method `__init__`, `__len__`, `num_writers`, `__getitem__` giống `TextDataset`, nhưng đọc nhánh `test`.

#### `TextCollator`

- `__init__(resolution)`: lưu resolution, dù `resolution` không được dùng trong collate.
- `__call__(batch)`: flatten nếu item là list (dùng cho sentence dataset), stack style tensors/writer IDs, pad ảnh thật theo width lớn nhất, và giữ label/z nếu có.

#### `CollectionTextDataset`

- `__init__`: cho phép `--dataset IAM,CVL`; tạo một dataset con từ mỗi `files/<name>-32.pickle`, hợp alphabet.
- `__len__`: tổng virtual length các dataset con.
- `num_writers`: tổng writer; lưu ý writer class ID của dataset sau chưa được offset.
- `__getitem__`: route global index vào dataset con.
- `get_dataset`: trả tên dataset chứa index.
- `collate_fn`: luôn dùng collator của dataset đầu; hợp lệ vì các collator hiện giống nhau.

#### Dataset phục vụ đánh giá

- `FidDataset.__init__`: đọc toàn bộ split train/test, tính tổng số sample và sort writer.
- `FidDataset.__len__`: số ảnh thật, không phải số writer.
- `FidDataset.num_writers`: số writer.
- `FidDataset.__getitem__`: chọn đúng sample theo global index, nhưng lấy N style reference ngẫu nhiên từ cùng writer; trả `img_id` và writer ID dạng int.
- `FidDatasetNum.__init__`: chỉ giữ sample có label hoàn toàn là số, nhưng style vẫn lấy từ tập gốc của writer.
- `FidDatasetNum.__getitem__`: triển khai lại logic cha để dùng `IMG_DATA_OLD` làm style pool.
- `FidDatasetLong.__init__`: chọn sample có ít nhất một ký tự hiếm hơn `threshold`; tên `Long` gây nhầm vì thực tế là long-tail character.
- `FidDatasetRand.__init__`: nạp word list ngoài.
- `FidDatasetRand.__getitem__`: thay label thật bằng một từ ngẫu nhiên trong word list để sinh text mới theo style thật.
- `FidDatasetApollo.__getitem__`: ép mọi label thành `Apollo11`.
- `FidDatasetAllNum.__len__`: cố định 10.000 sample.
- `FidDatasetAllNum.__getitem__`: dùng style/sample của lớp cha rồi đổi label thành số 4 chữ số theo index. Với index vượt độ dài dataset cha có thể lỗi.
- `TextSentence.__getitem__` / `TextSentenceval.__getitem__`: nhân một writer item thành bốn item cho câu “The eagle has landed”, dùng lần lượt style image làm placeholder `img`.
- `FolderDataset.__init__`: liệt kê toàn bộ file trực tiếp trong thư mục style.
- `FolderDataset.__len__`: số file.
- `FolderDataset.sample_style`: lấy N file **không hoàn lại**, resize cao 32 giữ tỉ lệ, pad/crop width 192 và trả `simg`, `swids`.

## 6. Mô hình sinh ảnh chính

### `models/model.py`

#### Hàm/lớp ở mức thấp

- `get_rgb(x)`: ánh xạ scalar 0..1 sang gradient màu xanh-đỏ; hiện không được gọi.
- `get_page_from_words(word_lists, MAX_IMG_WIDTH=800)`: ghép word images thành dòng/trang có gap 16 px. Có lỗi làm mất dòng cuối nếu trước đó đã wrap ít nhất một dòng.
- `FCNDecoder.__init__`: hai residual block, ba tầng nearest-neighbor upsample + conv/instance norm, cuối cùng conv+tanh ra một channel.
- `FCNDecoder.forward(x)`: chạy sequential decoder.

#### `Generator`

- `__init__(args)`: tạo ResNet-18 style encoder, Transformer encoder/decoder, Unifont query embedding, linear reshape, FCN decoder và các layer VAE tùy chọn. `avgpool`/`fc` của ResNet bị thay bằng Identity để giữ spatial tokens.
- `reparameterize(mu, logvar)`: sampling kiểu VAE theo từng token. Dùng `torch.cuda.FloatTensor` trực tiếp nên không chạy CPU; nhánh chỉ dùng khi `is_kld`.
- `reverse_forward(ST, QR)`: gần giống forward, đồng thời hook attention của layer cuối để lưu weight; tên “reverse” không phản ánh phép nghịch đảo nào.
- `Eval(ST, QRS, QRS_pos)`: encode style một lần, rồi sinh lần lượt từng word/query trong chiều thứ hai của `QRS`; trả list tensor ảnh. Có nhánh KLD và `all_chars` cũ.
- `compute_style(ST)`: ResNet encode N ảnh style, biến feature map thành sequence, chạy Transformer encoder và trả style memory.
- `forward(ST, QR, ...)`: training forward cho một batch word giả; Unifont query -> Transformer decoder -> linear/reshape -> FCN image. Các tham số `QRs`, `QR_pos`, `QRs_pos`, `mode` hiện không được sử dụng.

#### `VATr`

- `__init__(args)`: tạo `netG`, GAN discriminator `netD`, writer classifier `netW`, CRNN `netOCR`, Inception FID extractor, bốn Adam optimizer; nạp English lexicon và preview text. Việc dựng Inception có thể tải weight từ Internet ngay cả khi training không dùng FID.
- `save_images_for_fid_calculation(path, loader, split)`: sinh ảnh có cùng text/style với ảnh thật, lưu `Real_<split>/<writer>/<id>.png`, `Fake_<split>/...` và `ann.json`.
- `_generate_fakes(...)`: gọi `netG.Eval`, crop width đúng `length*resolution`, đổi `[-1,1]` về `[0,1]`, flatten kết quả theo batch rồi word.
- `_generate_page(ST, SLEN, ...)`: render trang gồm cột trái là style samples, cột phải là text sinh + bảng glyph đặc biệt; wrap theo `lwidth/rwidth`.
- `get_current_losses()`: trả dictionary các loss/regularizer hiện tại.
- `load_networks(epoch)`: API cũ dựa trên `BaseModel`, `self.opt`; không hoạt động trong class hiện tại.
- `_set_input(input)`: giữ batch để forward/optimize sau đó.
- `set_requires_grad(nets, requires_grad)`: bật/tắt gradient cho danh sách network.
- `forward()`: unpack batch; encode label thật; lấy word giả ngẫu nhiên từ lexicon; tạo one-hot; gọi generator bằng style và word giả.
- `backward_D_OCR()`: hinge loss discriminator trên real/fake + CTC OCR loss trên ảnh thật; backward tổng và dọn NaN/Inf gradient OCR.
- `backward_D_WL()`: hinge discriminator + cross-entropy nhận dạng writer trên ảnh thật.
- `optimize_D_WL()`: forward, bật D/W, zero grad, gọi `backward_D_WL` nhưng chưa step.
- `backward_D_OCR_WL()`: phiên bản legacy gộp D+OCR+writer, tham chiếu nhiều field không tồn tại (`self.opt`, `self.z`, `self.wcl`).
- `optimize_D_WL_step()`: step rồi zero optimizer D và writer.
- `backward_OCR()`: legacy OCR-only trên real; dùng `self.opt`, hiện lỗi.
- `backward_D()`: legacy D-only có latent `z`; hiện lỗi với pipeline hiện tại.
- `compute_cycle_loss()`: thay các style sample bằng ảnh fake lặp lại, re-encode style và tính L1 giữa style memory cũ/mới.
- `backward_G_only()`: loss G adversarial + tùy chọn OCR fake + cycle/KLD placeholder; cân bằng OCR/adversarial theo tỉ lệ độ lệch chuẩn gradient rồi backward.
- `backward_G_WL()`: tương tự nhưng dùng writer loss thay OCR loss.
- `backward_G()`: phiên bản legacy kết hợp adversarial/OCR/writer với `self.opt`; không thuộc train loop hiện tại.
- `optimize_D_OCR()`: forward; bật D/OCR; zero grad; gọi `backward_D_OCR`, chưa step.
- `optimize_OCR()`: legacy OCR-only, tham chiếu `self.opt.OCR_init`.
- `optimize_D()`: legacy D-only.
- `optimize_D_OCR_step()`: step/zero D và OCR.
- `optimize_D_OCR_WL()` / `optimize_D_OCR_WL_step()`: legacy pha gộp ba critic; dùng `self.opt`.
- `optimize_D_step()`: legacy step D và kiểm tra một cấu trúc `infer_img` không có trong `netD` hiện tại.
- `optimize_G()`: wrapper cho `backward_G` legacy.
- `optimize_G_WL()`: forward, freeze D/OCR/W, backward generator với writer objective.
- `optimize_G_only()`: forward, freeze critic, backward generator với OCR objective.
- `optimize_G_step()`: step rồi zero optimizer G.
- `optimize_ocr()`, `optimize_z()`, `optimize_parameters()`, `test()`, `train_GD()`, `save_networks()`: API legacy từ framework GAN cũ; phụ thuộc `self.opt`, `model_names`, `gpu_ids`, latent `z` hoặc `BaseModel`, không dùng được nguyên trạng.

### Bốn pha thật trong mỗi iteration của `train.py`

```text
1. optimize_G_only -> optimize_G_step
   G học realism + khả năng OCR đọc đúng nội dung.

2. optimize_D_OCR -> optimize_D_OCR_step
   D học phân biệt real/fake; OCR học đọc ảnh thật.

3. optimize_G_WL -> optimize_G_step
   G học realism + giữ đúng phong cách writer.

4. optimize_D_WL -> optimize_D_WL_step
   D học real/fake; W học phân loại writer từ ảnh thật.
```

Tần suất từng pha được điều khiển bởi bốn tham số `num_critic_*`. Mỗi wrapper gọi `forward()` riêng nên word giả được lấy lại ngẫu nhiên và generator chạy tối đa bốn lần cho một dataloader batch.

## 7. Discriminator, OCR và các building block

### `models/BigGAN_layers.py`

- `proj(x,y)`: projection vector dùng trong Gram-Schmidt.
- `gram_schmidt(x,ys)`: loại các thành phần của `x` theo các vector đã có.
- `power_iteration(W,u_,...)`: ước lượng singular value/vector lớn nhất, phục vụ spectral normalization.
- `identity.forward`: passthrough.
- `SN`: mixin đăng ký `u`, singular values và property `W_` trả weight đã chia singular value lớn nhất.
- `SNConv2d.forward`, `SNLinear.forward`, `SNEmbedding.forward`: convolution/linear/embedding dùng spectral-normalized weight.
- `Attention`: self-attention non-local kiểu SAGAN; `forward` tạo theta/phi/g, attention map rồi residual với hệ số `gamma` học được.
- `fused_bn`: áp normalization + gain/bias theo biểu thức gộp.
- `manual_bn`: tính mean/variance thủ công; tùy chọn trả statistics.
- `myBN.reset_stats`: xóa running/standing stats.
- `myBN.forward`: train bằng batch stats, eval bằng stored stats; hỗ trợ accumulate standing statistics.
- `groupnorm`: suy ra số group từ chuỗi `ch_*`, `grp_*`, hoặc mặc định 16.
- `ccbn`: conditional batch/instance/group norm; `forward` sinh gain/bias từ conditioning vector; `extra_repr` mô tả layer.
- `bn.forward`: batch norm không điều kiện, hỗ trợ synchronized/custom BN.
- `GBlock.forward`: residual generator block có conditional BN, activation, upsample và shortcut học được.
- `DBlock.shortcut`: downsample/project shortcut đúng thứ tự theo preactivation.
- `DBlock.forward`: residual discriminator block có spectral conv và tùy chọn average-pool downsample.

### `models/BigGAN_networks.py`

- `Decoder`: decoder cũ gần như trùng `FCNDecoder`; `forward` chạy sequential network.
- `G_arch(...)`: trả cấu hình kênh/upsample/attention cho nhiều resolution. Generator chính trong `models/model.py` không dùng class BigGAN Generator này.
- `Generator`: generator BigGAN/DETR legacy. `forward` gọi `self.DETR`, linear, reshape và `Decoder`; nhiều BigGAN block được dựng nhưng đã comment khỏi forward.
- `D_arch(...)`: bảng cấu hình discriminator theo resolution. Pipeline mặc định dùng key 16.
- `Discriminator.__init__`: dựng các spectral residual downsample block, optional attention, global linear score và projection embedding.
- `Discriminator.forward`: blocks -> ReLU -> global sum pool -> scalar critic; nếu có `y`, cộng projection class score.
- `Discriminator.return_features`: trả activation sau từng block.
- `WDiscriminator.__init__`: cùng backbone với D nhưng output dimension bằng số writer và dùng cross-entropy.
- `WDiscriminator.forward`: backbone -> global pool -> writer logits -> trả trực tiếp scalar cross-entropy với `y`.
- `WDiscriminator.return_features`: trả intermediate features.
- `Encoder`: kế thừa D, thay head bằng conv; `forward` trả latent feature map.
- `BiDiscriminator`: joint discriminator ảnh + latent; `forward` encode ảnh, broadcast latent theo width, concat rồi chấm điểm.
- `G_D`: wrapper legacy để chạy G và D hiệu quả trong DataParallel; `forward` hỗ trợ split/concat real-fake và trả optional generated image.

### `models/blocks.py`

- `ResBlocks`: xếp nhiều `ResBlock`; `forward` chạy toàn chuỗi.
- `ResBlock`: hai `Conv2dBlock` + residual addition.
- `ActFirstResBlock`: pre-activation residual block; `forward` dùng learned 1x1 shortcut nếu đổi kênh.
- `LinearBlock`: Linear + optional BN/IN + activation; `forward` theo đúng chuỗi đó.
- `Conv2dBlock`: reflection/replication/zero pad + Conv2d + optional BN/IN/AdaIN + activation; hỗ trợ activation-first.
- `AdaptiveInstanceNorm2d.forward`: reshape batch/channel rồi dùng `F.batch_norm` với weight/bias được gán từ ngoài.
- `AdaptiveInstanceNorm2d.__repr__`: tên layer và số feature.

### `models/OCR_network.py`

- `BidirectionalLSTM.__init__`: BiLSTM và Linear từ `2*nHidden` về `nOut`.
- `BidirectionalLSTM.forward`: sequence `[T,B,C]` -> BiLSTM -> projection mỗi timestep.
- `CRNN.__init__`: bảy convolution, pooling bất đối xứng để height còn 1; mặc định không dùng RNN mà Linear 512 -> vocab.
- `CRNN.forward`: CNN -> squeeze height -> permute thành `[T,B,vocab]`.
- `CRNN.backward_hook`: thay NaN trong input gradient bằng 0; hook mặc định không được đăng ký.
- `OCRLabelConverter`: converter CTC dạng targets nối phẳng; `encode` bytes -> 1-based IDs và lengths; `decode` loại blank và ký tự lặp.
- `strLabelConverter`: converter pipeline đang dùng; `encode` trả padded `[B,L]`, lengths và `None`; `decode` hiện vẫn viết theo định dạng target nối phẳng nên không khớp trực tiếp output padded.
- `UnifontLabelConverter.__init__`: đọc JSON component proposal cho từng ký tự và tạo vocabulary template.
- `UnifontLabelConverter.encode`: biến text thành chuỗi component-template IDs + center coordinates.
- `UnifontLabelConverter.decode`: chưa triển khai.

## 8. Transformer, Unifont và positional encoding

### `models/transformer.py`

Đây là Transformer fork từ DETR, nhận positional/query embeddings trực tiếp trong attention.

- `Transformer.__init__`: dựng encoder/decoder clone layers; `_reset_parameters` Xavier-init mọi tensor >1D.
- `Transformer.forward`: flatten feature map, lookup query, decode và concat decoder feature với query embedding. Class này không phải class trực tiếp dùng trong `models/model.py`; pipeline dùng các lớp encoder/decoder con.
- `TransformerEncoder.forward`: chạy tuần tự encoder layers rồi optional final norm.
- `TransformerDecoder.forward`: chạy decoder layers, optional lưu output từng layer, final norm, trả stack `[layers,...]`.
- `TransformerEncoderLayer.with_pos_embed`: cộng positional embedding nếu có.
- `forward_post` / `forward_pre`: hai thứ tự post-norm hoặc pre-norm cho self-attention + FFN + residual.
- `TransformerEncoderLayer.forward`: chọn pre/post theo `normalize_before`.
- `TransformerDecoderLayer.with_pos_embed`: cộng query/position.
- `forward_post` / `forward_pre`: self-attention trên query, cross-attention với style memory, FFN và residual theo post/pre norm.
- `TransformerDecoderLayer.forward`: dispatch pre/post norm.
- `_get_clones(module,N)`: deep-copy layer N lần.
- `build_transformer(args)`: factory dùng tên option kiểu DETR cũ (`hidden_dim`, `nheads`...).
- `_get_activation_fn`: map `relu`, `gelu`, `glu` sang hàm activation.

### `models/unifont_module.py`

- `UnifontModule.__init__`: nạp cả glyph gốc `unifont` và representation theo `input_type`; thêm vector zero ở index 0; optional Linear về `out_dim`.
- `get_symbols(input_type)`: đọc `files/<input_type>.pickle`, map Unicode codepoint -> bitmap flatten, chọn đúng alphabet theo thứ tự.
- `forward(QR)`: index bitmap bằng ID text rồi project qua Linear/Identity.
- `LearnableModule`: query embedding thay thế dùng một parameter 256 chiều; `forward` project rồi repeat theo batch. Hiện không được chọn.

### `models/unifont_symbol.py`

- `random_color`, `hex2int`: tiện ích màu để visualize graph.
- `image_grid`: ghép PIL images vào grid, pad bằng ảnh rỗng.
- `add_margin`: thêm viền cho PIL image.
- `Node`: pixel-node có tập connection; `connect` nối hai chiều; `cdist` khoảng cách Euclid bình phương; `sdist` Chebyshev; `draw_connections` vẽ cạnh/node; `__str__`/`__repr__` mô tả node.
- `Symbol.__init__`: tạo bitmap symbol từ path hoặc matrix.
- `make_graph`: mỗi foreground pixel thành node và nối 8-neighborhood.
- `endpoints`: node có degree <2.
- `center_of_mass`: trung bình tọa độ foreground.
- `components`: tách connected components bằng OpenCV.
- `rotate`, `fliplr`, `flipud`: biến đổi hình học, trả Symbol mới.
- `match`: đếm số vị trí template xuất hiện chính xác trong symbol khác.
- `locate`: trả các bản symbol được pad theo vị trí match.
- `__contains__`: kiểm tra chứa template.
- `pad`: pad top/left theo tọa độ.
- `jointable`: kiểm tra endpoint hai symbol cách nhau <2.
- `sum`, `__lt__`, `__le__`, `__eq__`: số pixel và so sánh pixel-count, size hoặc equality của bitmap.
- `__hash__`: hash bitmap đã pad 16x16; `__repr__`: shape + pixel count.
- `__add__`, `__sub__`, `__mul__`: OR, hiệu logic, AND hai bitmap.
- `squeeze`: crop bounding box; `unsqueeze`: pad về 16x16.
- `exp_img`, `show`, `save`: phóng to/vẽ graph/hiển thị/lưu.
- `is_special`: nhận diện header bit pattern đặc biệt.
- `toJSON` / `fromJSON`: serialize/deserialize matrix và indices.
- `EmptySymbol.squeeze`: giữ nguyên symbol rỗng.
- `divide_chunks`: yield list theo chunk size.
- `TemplateSet.add`: deduplicate theo hash và hợp các index; `__iter__`, `__len__` cung cấp collection API.
- Main block: công cụ offline để tách component glyph, augment rotation/flip và xuất template JSON; dùng path `imgs/unifont-14.0.04` không có trong repo.

### `models/positional_encodings.py`

- `get_emb`: xen kẽ sin/cos cho input frequency.
- `PositionalEncoding1D`, `PositionalEncoding2D`, `PositionalEncoding3D`: constructor tạo inverse frequencies; `forward` kiểm tra rank, tạo/cached sinusoidal encoding đúng shape.
- `PositionalEncodingPermute1D`, `PositionalEncodingPermute2D`, `PositionalEncodingPermute3D`: adapter cho tensor channel-first; `org_channels` forward metadata từ encoder trong.
- `Summer.forward`: cộng encoding vào tensor sau khi kiểm tra shape.
- `SparsePositionalEncoding2D.__init__`: tạo canvas encoding cố định.
- `SparsePositionalEncoding2D.forward`: pad list coordinate, chuẩn hóa về grid `[-1,1]`, lấy encoding bằng `grid_sample`.

## 9. FID/Inception và synchronized batch norm

### `models/inception.py`

- `InceptionV3.__init__`: chia Inception v3 thành bốn block output 64/192/768/2048 chiều; mặc định tải bản weight tương thích TensorFlow FID và freeze parameter.
- `InceptionV3.forward`: resize 299x299, normalize `[0,1]` -> `[-1,1]`, chạy đến block cần thiết và trả list feature maps.
- `fid_inception_v3`: dựng torchvision Inception, thay các mixed block khác biệt và tải FID weights.
- `FIDInceptionA.forward`, `FIDInceptionC.forward`, `FIDInceptionE_1.forward`: triển khai branch như torchvision nhưng average pool không tính padded zeros.
- `FIDInceptionE_2.forward`: tương tự nhưng branch pool dùng max pooling đúng bản FID weight.
- Hàm tính Frechet distance có trong file nhưng bị comment, nên module này chỉ trích feature chứ không tính FID cuối cùng.

### `models/sync_batchnorm/`

`batchnorm.py`:

- `_sum_ft`, `_unsqueeze_ft`: reduce/restore các dimension ngoài channel.
- `_SynchronizedBatchNorm.forward`: dùng BN thường khi single-device/eval; khi DataParallel thì gửi sum/square-sum về master, nhận global mean/inv-std rồi normalize.
- `__data_parallel_replicate__`: gán vai master/slave khi model được replicate.
- `_data_parallel_master`: reduce statistics từ GPU, tính stats, broadcast kết quả.
- `_compute_mean_std`: mean, biased/unbiased variance, cập nhật running stats.
- `SynchronizedBatchNorm1d`, `SynchronizedBatchNorm2d`, `SynchronizedBatchNorm3d`: ba biến thể theo rank tensor; method `_check_input_dim` validate rank đầu vào.

`comm.py`:

- `FutureResult.put/get`: future một phần tử thread-safe.
- `SlavePipe.run_slave`: gửi message lên queue, chờ master, acknowledge.
- `SyncMaster.__getstate__` / `__setstate__`: chỉ serialize master callback rồi dựng lại queue/registry khi unpickle.
- `SyncMaster.register_slave`: tạo pipe cho replica.
- `SyncMaster.run_master`: gom message, gọi callback, trả kết quả từng slave.
- `nr_slaves`: số replica đã đăng ký.

`replicate.py`:

- `CallbackContext`: object rỗng dùng chia sẻ state giữa replicas.
- `execute_replication_callbacks`: gọi hook của từng submodule theo thứ tự master trước.
- `DataParallelWithCallback.replicate`: replicate rồi chạy callback.
- `patch_replication_callback`: monkey-patch một `DataParallel` có sẵn; nested `new_replicate` gọi replicate gốc rồi callback.

`batchnorm_reimpl.py`:

- `BatchNorm2dReimpl`: bản BN thủ công cho test numerical stability; `reset_running_stats` đặt mean=0/variance=1, `reset_parameters` reset stats và affine, `forward` tự tính mean/variance/affine.

`unittest.py`:

- `TorchTestCase.assertTensorClose`: assertion `torch.allclose` kèm absolute/relative difference. File không chứa test case cụ thể.

`__init__.py` re-export ba SyncBN class và hai helper DataParallel.

## 10. Utility modules và entry points

### `train.py`

- `main()`: parse hyperparameters; seed; tạo train/test dataset + DataLoader; suy ra alphabet/writer count; dựng VATr; resume checkpoint hoặc load ResNet; optional W&B; chạy bốn pha train; mỗi epoch render preview, log loss, lưu latest và history checkpoint.
- `rSeed(sd)`: seed Python, NumPy, torch CPU và CUDA.
- Exception wrapper gọi `wandb.finish()` rồi re-raise.

Checkpoint train có dạng:

```python
{"model": model.state_dict(), "wandb_id": id, "epoch": epoch}
```

### `generator.py`

- `load_checkpoint(model, checkpoint)`: nhận state dict hoặc checkpoint có key `model`; hỗ trợ format cũ đúng 241 keys bằng cách map prefix `netG.` và map riêng ResNet; format khác map theo thứ tự key/shape.
- `FakeArgs`: cấu hình inference tương thích constructor `VATr`.
- `VATr_writer.__init__`: dựng VATr, load checkpoint, chuyển eval.
- `set_style_folder`: tạo `FolderDataset` từ ảnh mẫu.
- `generate(texts)`: chuẩn hóa string thành list; với mỗi text lấy ngẫu nhiên 15 style images, encode từng word, sinh từng ảnh word, ghép ngang bằng gap 16 px, trả list NumPy uint8.
- Main CLI: nhận style folder/text/text-file/checkpoint/output/noise, tạo thư mục output và lưu mỗi input line thành `<stem>_NNN.<ext>`.

### `generate_fakes.py`

- `load_checkpoint`: cùng ý tưởng như bản trong `generator.py`, nhưng caller đã bóc key `model` trước.
- `filter_nums(loader)`: generator cục bộ lọc item label numeric trong batch; hiện không được gọi.
- Main: parse config, chọn pickle IAM/CVL, tạo `FidDataset` train/test, load model/checkpoint, gọi `save_images_for_fid_calculation` cho cả hai split.

### `models/networks.py`

- `init_weights`: apply N(0,.02), Xavier, Kaiming hoặc Orthogonal cho Conv/Linear/Embedding; nếu tên khác thì coi là path/model initialization legacy.
- nested `init_func`: callback thực hiện initialization từng module.
- `init_net`: move/DataParallel rồi init weight.
- `get_scheduler`: tạo linear/step/plateau/cosine LR scheduler từ option object kiểu CycleGAN cũ.

### `models/__init__.py`

- `find_model_using_name`: dynamic import `models/<name>_model.py` và tìm class theo convention.
- `get_option_setter`: lấy static option modifier của model class.
- `create_model`: instantiate model động.

Ba hàm này là code framework cũ và hiện lỗi vì `BaseModel` không được import/định nghĩa; pipeline hiện tại import trực tiếp `VATr` nên không đi qua đây.

### `util/util.py`

- `random_word`: sinh chuỗi ký tự ngẫu nhiên.
- `load_network`: load `<epoch>_net_<name>.pth` vào network.
- `writeCache`: ghi dictionary vào LMDB.
- `loadData`: resize/copy tensor không theo gradient.
- `multiple_replace`: replace tuần tự theo dictionary.
- `get_curr_data`: cắt một mini-batch con theo counter.
- `seed_rng`: seed torch CPU/CUDA và NumPy.
- `make_one_hot`: padded character IDs -> `[B,L,C]` one-hot, bỏ offset 1-based.
- `loss_hinge_dis`: hinge loss real/fake; optional mask vùng sau độ dài text đối với output spatial.
- `loss_hinge_gen`: generator hinge loss âm trung bình, optional mask.
- `loss_std`: độ lệch chuẩn latent theo width và latent mean.
- `toggle_grad`: bật/tắt requires-grad; hàm bị định nghĩa trùng hai lần, bản sau ghi đè bản trước nhưng logic giống nhau.
- `ortho`: cộng modified orthogonal regularization trực tiếp vào gradient, bỏ diagonal.
- `default_ortho`: orthogonal regularization chuẩn vào gradient.
- `Distribution.init_distribution`: gắn metadata normal/categorical/poisson/gamma và seed.
- `Distribution.sample_`: sample in-place theo loại phân phối.
- `Distribution.to`: move/cast nhưng giữ wrapper metadata.
- `to_device`: move network và optional DataParallel nhiều GPU.
- `prepare_z_y`: tạo latent normal `z` và class categorical `y` dạng `Distribution`.
- `tensor2im`: tensor `[-1,1]` -> RGB uint8 NumPy.
- `diagnose_network`: in trung bình trị tuyệt đối gradient.
- `save_image`: NumPy -> PIL -> file.
- `print_numpy`: in statistics/shape.
- `mkdirs`, `mkdir`: tạo một hoặc nhiều thư mục nếu chưa tồn tại.

### `util/misc.py`

- `SmoothedValue`: giữ cửa sổ metric; `update`, `synchronize_between_processes`, và các property median/avg/global_avg/max/value; `__str__` format thống kê.
- `all_gather`: pickle object bất kỳ thành byte tensor, pad rồi all-gather giữa ranks.
- `reduce_dict`: sum/average dictionary tensor giữa ranks.
- `MetricLogger`: `update`; `__getattr__` cho phép truy cập meter như attribute; `__str__` format; `synchronize_between_processes`; `add_meter`; `log_every` yield iterable và định kỳ in ETA/time/GPU memory.
- `get_sha`: lấy git SHA, branch và dirty state qua subprocess; nested `_run` chạy git command.
- `collate_fn`: transpose batch bằng `zip(*batch)`.
- `_max_by_axis`: max từng trục của list shape.
- `NestedTensor`: giữ padded tensor + mask; `to`, `decompose`, `__repr__`.
- `nested_tensor_from_tensor_list`: pad list ảnh 3D thành batch và mask; dispatch bản ONNX khi tracing.
- `_onnx_nested_tensor_from_tensor_list`: phiên bản dùng `F.pad` tương thích ONNX.
- `setup_for_distributed`: thay built-in print để chỉ master in; nested `print` hỗ trợ `force=True`.
- `is_dist_avail_and_initialized`, `get_world_size`, `get_rank`, `is_main_process`, `save_on_master`: helper distributed.
- `init_distributed_mode`: đọc env RANK/WORLD_SIZE hoặc SLURM, chọn GPU/NCCL và init process group.
- `accuracy`: top-k accuracy.
- `interpolate`: compatibility wrapper cho torchvision cũ và empty tensor.

### `util/html.py`

- `HTML.__init__`: tạo web/images dirs và DOM document, optional meta refresh.
- `get_image_dir`: trả thư mục ảnh.
- `add_header`: thêm heading.
- `add_images`: thêm một table row ảnh/caption/link.
- `save`: render `index.html`.
- Main block minh họa cách dùng.

### `util/visualizer.py`

- `save_images`: chuyển tensor thành ảnh, optional resize theo aspect ratio, lưu và thêm vào HTML.
- `Visualizer.__init__`: kết nối Visdom, tạo HTML/log dirs và loss log.
- `reset`: reset cờ đã lưu.
- `create_visdom_connections`: spawn Visdom server khi kết nối thất bại.
- `display_current_results`: hiển thị grid/panel Visdom và lưu lịch sử HTML theo epoch.
- `plot_current_losses`: tích lũy và vẽ đường loss.
- `print_current_losses`: in và append loss/time vào log.

`util/__init__.py` chỉ chứa copyright header, không export logic.

## 11. Các vấn đề quan trọng phát hiện được

### P0 — chặn chạy ngay

1. **Thiếu dataset và checkpoint:** không có `IAM-32.pickle`, `CVL-32.pickle`, `vatr.pth`, `resnet_18_pretrained.pth`.
2. **README không khớp code:** README yêu cầu `scripts/prepare_iam.py` và tạo JSONL, nhưng script không tồn tại; loader chỉ đọc pickle.
3. **`wandb` được import bắt buộc:** `train.py` luôn `import wandb`, dù requirements xếp nó là optional. Không cài optional package thì train chết trước khi parse `--wandb`.
4. **`UnifontModule` mặc định ép glyph lên CUDA:** `Generator` không truyền `args.device`, vì vậy `generator.py` tuy chọn CPU khi không có GPU vẫn lỗi lúc dựng model.
5. **Dependency chuẩn bị/visualize chưa khai báo:** `lmdb`, `xmltodict`, `scikit-learn`, `dominate`, `visdom` không nằm trong requirements. `visualizer.py` còn import `scipy.misc.imresize`, API đã bị xóa khỏi SciPy mới.

### P1 — lỗi runtime/logic ở đường đang dùng hoặc gần đường đang dùng

1. `VATr.__init__` luôn dựng Inception FID và tải weights qua mạng dù training/inference thường không dùng; offline sẽ lỗi hoặc khởi động rất chậm.
2. `save_images_for_fid_calculation` gọi `json.dump` nhưng `models/model.py` không import `json`.
3. CRNN trả raw logits cho `torch.nn.CTCLoss`; PyTorch yêu cầu **log-probabilities** (thường `log_softmax(dim=2)`), nên loss hiện không đúng hợp đồng toán học.
4. `get_page_from_words` không append phần `line_t` còn lại sau lần wrap đầu tiên.
5. `FolderDataset.sample_style(..., replace=False)` lỗi nếu folder có ít hơn `num_examples` ảnh.
6. `train.py` mặc định `epochs=100000` và lưu/render theo epoch; mỗi epoch chỉ có xấp xỉ số writer/batch iteration, cần hiểu đây là convention chứ không phải epoch dataset chuẩn.
7. `CollectionTextDataset` cộng số writer nhưng không offset `wcl` giữa nhiều dataset, nên writer classifier bị trùng class ID nếu train `IAM,CVL`.
8. `torch.load()` không truyền `map_location` trong train/generate_fakes; checkpoint GPU có thể không load trên CPU/máy khác.
9. `ResNet18(weights=DEFAULT)` đã tự tải pretrained weights, rồi `train.py` lại yêu cầu load `resnet_18_pretrained.pth`; hai cơ chế pretrain chồng nhau.

### P2 — mã legacy/dead code và khả năng bảo trì

1. Nhiều method trong `VATr` tham chiếu `self.opt`, `BaseModel`, `self.z`, `self.real_z_mean`, `self.model_names`, `self.gpu_ids` không tồn tại.
2. `models/__init__.py` tham chiếu `BaseModel` không được import.
3. BigGAN `Generator`, `Encoder`, `BiDiscriminator`, `G_D`, AdaIN, positional encoding và phần lớn distributed utilities không nằm trong train loop hiện tại.
4. `data/create_data.py` và main của `prepare_data.py` chứa đường dẫn NFS của máy tác giả.
5. Các boolean argparse dùng `type=bool` (`--is_seq`, `--is_kld`,...) không parse CLI như mong đợi: chuỗi `False` vẫn truthy. Nên dùng `store_true/store_false`.
6. Alphabet/special alphabet hiển thị mojibake trong source/README ở môi trường đọc hiện tại; cần chuẩn hóa UTF-8 để Greek characters và tiếng Việt không bị sai.
7. `strLabelConverter.decode` được kế thừa từ converter targets nối phẳng nhưng `encode` lại trả padded 2D tensor.
8. `FidDatasetAllNum` có thể gọi lớp cha với index lớn hơn dataset thật.
9. `createDataset` gán `discard_wide=False` bên trong, vô hiệu hóa đối số cùng tên; suffix output cho `discard_narr` cũng ghi nhầm `_NoDiscard_wide`.
10. Trong SyncBN non-parallel, code cộng cả `gain` và `bias` thay vì áp scale rồi shift; nhánh này không được pipeline mặc định dùng.

## 12. Trình tự chạy dự kiến sau khi hoàn thiện dữ liệu

### Huấn luyện

```powershell
python train.py --dataset IAM --device cuda --batch_size 8 --num_examples 15
```

Điều kiện cần: có `files/IAM-32.pickle`, Unifont đúng alphabet, GPU/CUDA, các dependency, và xử lý lại cơ chế ResNet/Inception weights.

Output checkpoint:

```text
saved_models/<experiment-name>/model.pth
saved_models/<experiment-name>/<epoch>_model.pth
```

### Inference theo style folder

```powershell
python generator.py `
  --style-folder files/style_samples/00 `
  --text "Hello world" `
  --checkpoint <checkpoint.pth> `
  --output outputs/result.png
```

Output thực tế sẽ là `outputs/result_000.png`, không phải đúng tên `result.png`.

### Xuất tập FID

```powershell
python generate_fakes.py --checkpoint <checkpoint.pth> --dataset IAM --output saved_images
```

Output:

```text
saved_images/<checkpoint-stem>/
|-- Real_train/<writer>/*.png
|-- Fake_train/<writer>/*.png
|-- Real_test/<writer>/*.png
|-- Fake_test/<writer>/*.png
`-- ann.json
```

Script này chỉ xuất ảnh/annotation; không tính FID score.

## 13. Phân loại phần cốt lõi và phần có thể tách bỏ

**Cốt lõi đang dùng:** `train.py`, `generator.py`, `generate_fakes.py`, `data/dataset.py`, `models/model.py`, `models/transformer.py`, `models/unifont_module.py`, `models/OCR_network.py`, `models/BigGAN_networks.py` (D/W), `models/BigGAN_layers.py`, `models/blocks.py`, `util/util.py`, `files/unifont.pickle`, lexicon và style samples.

**Chỉ dùng cho đánh giá/preview:** `models/inception.py`, `mytext.txt`, corpora, Groundtruth, output samples, model diagrams.

**Legacy hoặc hiện không nối vào pipeline chính:** `data/create_data.py`, phần lớn `data/prepare_data.py`, `dataset/`, BigGAN Generator/BiDiscriminator/G_D, `models/networks.py` scheduler, `models/__init__.py` dynamic factory, positional encodings, Unifont symbol decomposition, SyncBN, `util/misc.py`, `util/html.py`, `util/visualizer.py`.

Điểm phân loại “legacy” không có nghĩa là thuật toán vô dụng; nó nghĩa là không có đường gọi từ ba entry point hiện tại hoặc phụ thuộc interface đã bị loại bỏ.
