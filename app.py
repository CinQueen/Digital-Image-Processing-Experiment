from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for, session, jsonify
import os
from utils.basic_ops import process_image_basic
from utils.enhance_ops import enhance_image, hide_image  
from werkzeug.utils import secure_filename
from PIL import Image
from utils.enhance_ops import smooth_image  
from utils.enhance_ops import (
    roberts_operator,
    sobel_operator,
    laplacian_operator,
    prewitt_operator,
    apply_ordered_filters,
    hist_eq
) 
from utils.deep_learning_ops import style_transfer, face_beauty_gfpgan, deblurgan_deblur  
from utils.frequency_ops import (
    fft_transform,
    apply_lowpass_filters,
    apply_highpass_filters
)
from utils.segmentation_ops import otsu_segmentation, improved_canny_edge_detection
from utils.deep_segmentation import mask_rcnn_segmentation, unet_segmentation
from utils.feature_ops import sift_detection, orb_detection, image_stitching
from utils.fusion_ops import image_fusion
from utils.multiband_fusion import MultiBandFusion
from utils.superglue_matcher import SuperGlueMatcher
from utils.deep_dehaze import cyclegan_dehaze  
from utils.restoration_ops import (
    dark_channel_dehazing,
    deblur_image,
    inpaint_photo_restoration,
    color_correction_with_histogram_matching,
    restore_old_photo
)

# 确保上传文件夹存在
os.makedirs('static/uploads', exist_ok=True)

# 设置文件权限（如果在Linux/Unix系统上）
try:
    os.chmod('static/uploads', 0o777)
except Exception as e:
    print(f"设置文件夹权限时出错: {str(e)}")

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于闪存消息
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# 初始化 SuperGlue 匹配器
superglue_matcher = SuperGlueMatcher()

@app.route('/', methods=['GET', 'POST'])
def index():
    processed_image = None
    processed_images = None  # 用于存储多个滤波结果
    uploaded_file = session.get('uploaded_file')  # 获取会话中的上传文件路径
    secret_file = session.get('secret_file')  # 获取会话中的秘密图像路径

    if request.method == 'POST':
        # 检查是否上传了文件
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['uploaded_file'] = filepath  # 保存上传文件路径到会话
            uploaded_file = filepath  # 更新当前文件
            flash("文件上传成功！")
        elif not uploaded_file:
            flash("未找到文件，请重新上传！")
            return redirect(request.url)

        # 检查是否上传了秘密图像
        if 'secret_image' in request.files and request.files['secret_image'].filename != '':
            secret = request.files['secret_image']
            secret_filename = secure_filename(secret.filename)
            secret_filepath = os.path.join(app.config['UPLOAD_FOLDER'], secret_filename)
            secret.save(secret_filepath)
            session['secret_file'] = secret_filepath  # 保存秘密图像路径到会话
            secret_file = secret_filepath  # 更新秘密图像路径
            flash("秘密图像上传成功！")

        operation = request.form.get('operation')

        # 检查是否选择了操作类型
        if not operation:
            flash("未选择操作类型，请选择后重试！")
            return redirect(request.url)
        
        if operation == 'crop':
            # 如果是裁剪操作，直接渲染裁剪界面
            return render_template(
                'index.html',
                uploaded_file=uploaded_file,
                operation=operation
            )
            
        try:
            # 根据操作类型处理图像
            if operation in ['read', 'grayscale', 'binary']:
                processed_image = process_image_basic(uploaded_file, operation)
            elif operation in ['log', 'exp', 'linear']:
                processed_image = enhance_image(uploaded_file, operation)
            elif operation == 'smooth':
                # 获取用户选择的高斯核大小
                kernel_size = int(request.form.get('kernel_size', None))
                print(f"Received kernel_size from form: {kernel_size}")  # 调试信息
                if kernel_size % 2 == 0:
                    flash("核大小必须为奇数，请重新输入！")
                    return redirect(request.url)
                processed_image = smooth_image(uploaded_file, kernel_size)

                # 转换为相对路径以供模板使用
                processed_image = os.path.relpath(processed_image, 'static')
                processed_image = processed_image.replace("\\", "/")  # 替换反斜杠为正斜杠
                processed_image = url_for('static', filename=processed_image)

                # 渲染模板时传递核大小
                return render_template(
                    'index.html',
                    processed_image=processed_image,
                    uploaded_file=uploaded_file,
                    operation=operation,
                    kernel_size=kernel_size
                )
            elif operation == 'hist_eq':
                method = request.form.get('hist_method', 'global')  # 获取均衡化方法
                original_image, processed_image = hist_eq(uploaded_file, method)
                original_image = os.path.relpath(original_image, 'static').replace("\\", "/")
                processed_image = os.path.relpath(processed_image, 'static').replace("\\", "/")
                original_image = url_for('static', filename=original_image)
                processed_image = url_for('static', filename=processed_image)
                return render_template(
                    'index.html',
                    processed_image=processed_image,
                    uploaded_file=uploaded_file,
                    original_image=original_image,
                    operation=operation,
                    hist_method=method
                )
            elif operation == 'hide':
                if not secret_file:
                    flash("请上传秘密图像后再尝试隐藏操作！")
                    return redirect(request.url)
                processed_image = hide_image(uploaded_file, secret_file)

                # 转换为相对路径以供模板使用
                processed_image = os.path.relpath(processed_image, 'static')
                processed_image = processed_image.replace("\\", "/")  # 替换反斜杠为正斜杠
                processed_image = url_for('static', filename=processed_image)

                return render_template(
                    'index.html',
                    processed_image=processed_image,
                    uploaded_file=uploaded_file,
                    operation=operation
                )
            elif operation == 'roberts':
                processed_image = roberts_operator(uploaded_file)
            elif operation == 'sobel':
                processed_image = sobel_operator(uploaded_file)
            elif operation == 'laplacian':
                processed_image = laplacian_operator(uploaded_file)
            elif operation == 'prewitt':
                processed_image = prewitt_operator(uploaded_file)
            elif operation == 'highpass':
                cutoff = int(request.form.get('cutoff', 30))
                filter_results = apply_highpass_filters(uploaded_file, cutoff)
                
                # 转换所有结果路径为相对路径
                processed_images = {
                    key: url_for('static', filename=os.path.relpath(path, 'static').replace("\\", "/"))
                    for key, path in filter_results.items()
                }
                
                return render_template(
                    'index.html',
                    processed_images=processed_images,
                    uploaded_file=uploaded_file,
                    operation=operation,
                    cutoff=cutoff
                )
            elif operation == 'sequence_filter':
                # 获取用户选择的核大小
                kernel_size = int(request.form.get('filter_kernel_size', 3))
                if kernel_size % 2 == 0:
                    flash("核大小必须为奇数，请重新输入！")
                    return redirect(request.url)
                # 应用顺序性滤波
                filter_results = apply_ordered_filters(uploaded_file, kernel_size)
                processed_images = {
                    key: url_for('static', filename=os.path.relpath(path, 'static').replace("\\", "/"))
                    for key, path in filter_results.items()
                }
                return render_template(
                    'index.html',
                    processed_images=processed_images,
                    uploaded_file=uploaded_file,
                    operation=operation
                )
            elif operation == 'beauty':
                # 添加进度显示
                session['processing_progress'] = 0
                session['processing_stage'] = "准备处理"
                def progress_callback(stage, progress):
                    session['processing_stage'] = stage
                    session['processing_progress'] = progress
                    
                processed_image = face_beauty_gfpgan(uploaded_file, progress_callback)
            elif operation == 'style_transfer':
                style_type = request.form.get('style_type', 'starry_night')
                processed_image = style_transfer(uploaded_file, style_type)
            elif operation == 'fft':
                results = fft_transform(uploaded_file)
                return render_template(
                    'index.html',
                    fft_results=results,
                    uploaded_file=uploaded_file,
                    operation=operation
                )
            elif operation == 'lowpass':
                cutoff = int(request.form.get('cutoff', 30))
                filter_results = apply_lowpass_filters(uploaded_file, cutoff)
                
                # 转换所有结果路径为相对路径
                processed_images = {
                    key: url_for('static', filename=os.path.relpath(path, 'static').replace("\\", "/"))
                    for key, path in filter_results.items()
                }
                
                return render_template(
                    'index.html',
                    processed_images=processed_images,
                    uploaded_file=uploaded_file,
                    operation=operation,
                    cutoff=cutoff
                )
            elif operation == 'otsu':
                processed_image = otsu_segmentation(uploaded_file)
            elif operation == 'canny':
                # 获取Canny参数
                low_threshold = int(request.form.get('low_threshold', 100))
                high_threshold = int(request.form.get('high_threshold', 200))
                sigma = float(request.form.get('sigma', 1.0))
                
                edge_results = improved_canny_edge_detection(
                    uploaded_file, 
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    sigma=sigma
                )
                
                # 转换路径格式
                processed_images = {
                    key: url_for('static', filename=os.path.relpath(path, 'static').replace("\\", "/"))
                    for key, path in edge_results.items()
                }
                
                return render_template(
                    'index.html',
                    processed_images=processed_images,
                    uploaded_file=uploaded_file,
                    operation=operation,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    sigma=sigma
                )
            elif operation == 'maskrcnn':
                confidence = float(request.form.get('confidence', 0.5))
                processed_image = mask_rcnn_segmentation(uploaded_file, confidence_threshold=confidence)
                processed_image = os.path.relpath(processed_image, 'static').replace("\\", "/")
                processed_image = url_for('static', filename=processed_image)
            elif operation == 'unet':
                try:
                    processed_image = unet_segmentation(uploaded_file)
                    # 确保路径转换正确
                    processed_image = os.path.relpath(processed_image, 'static').replace("\\", "/")
                    processed_image = url_for('static', filename=processed_image)
                    
                    return render_template(
                        'index.html',
                        processed_image=processed_image,
                        uploaded_file=uploaded_file,
                        operation=operation
                    )
                except Exception as e:
                    print(f"U-Net处理错误: {str(e)}")  # 添加错误日志
                    flash(f"处理图像时出错：{str(e)}")
                    return redirect(request.url)
            elif operation == 'sift_detect':
                processed_image = sift_detection(uploaded_file)
            elif operation == 'orb_detect':
                processed_image = orb_detection(uploaded_file)
            elif operation == 'image_stitch':
                try:
                    print("\n=== 图像拼接处理开始 ===")
                    
                    if 's_second_image' not in request.files:
                        print("未找到第二张图片")
                        flash("请选择第二张图片进行拼接！")
                        return redirect(request.url)
                        
                    second_file = request.files['s_second_image']
                    if second_file.filename == '':
                        print("第二张图片文件名为空")
                        flash("请选择第二张图片！")
                        return redirect(request.url)
                        
                    stitch_method = request.form.get('stitch_method', 'sift')
                    print(f"使用的特征点检测方法: {stitch_method}")
                    
                    # 保存第二张图片
                    second_filename = secure_filename(second_file.filename)
                    second_filepath = os.path.join(app.config['UPLOAD_FOLDER'], second_filename)
                    second_file.save(second_filepath)
                    print(f"第二张图片已保存到: {second_filepath}")
                    
                    try:
                        # 进行图像拼接
                        processed_image = image_stitching(
                            uploaded_file, 
                            second_filepath, 
                            method=stitch_method
                        )
                        
                        print(f"拼接结果保存在: {processed_image}")
                        
                        # 转换路径为URL
                        processed_image = os.path.relpath(processed_image, 'static')
                        processed_image = processed_image.replace("\\", "/")
                        processed_image = url_for('static', filename=processed_image)
                        
                        return render_template('index.html',
                                            processed_image=processed_image,
                                            uploaded_file=uploaded_file,
                                            operation='image_stitch',
                                            stitch_method=stitch_method)
                            
                    except Exception as e:
                        print(f"图像拼接错误: {str(e)}")
                        flash(f"图像拼接失败：{str(e)}")
                        return redirect(request.url)
                        
                except Exception as e:
                    print(f"处理请求错误: {str(e)}")
                    flash(f"处理失败：{str(e)}")
                    return redirect(request.url)
            elif operation == 'image_fusion':
                if 'fusion_image' not in request.files:
                    flash("请选择要融合的图片！")
                    return redirect(request.url)
                    
                fusion_file = request.files['fusion_image']
                fusion_method = request.form.get('fusion_method', 'wavelet')
                
                # 保存融合图片
                fusion_filename = secure_filename(fusion_file.filename)
                fusion_filepath = os.path.join(app.config['UPLOAD_FOLDER'], fusion_filename)
                fusion_file.save(fusion_filepath)
                
                gray_path, colored_path = image_fusion(uploaded_file, fusion_filepath, method=fusion_method)
                
                # 转换路径为URL
                colored_url = url_for('static', filename=os.path.relpath(colored_path, 'static').replace('\\', '/'))
                
                return render_template('index.html',
                                     uploaded_file=uploaded_file,
                                     processed_image=colored_url,
                                     operation='image_fusion')
            elif operation == 'multiband_fusion':
                if 'band1' not in request.files or 'band2' not in request.files:
                    flash("请选择两个波段的图像！")
                    return redirect(request.url)
                    
                band1_file = request.files['band1']
                band2_file = request.files['band2']
                
                if band1_file.filename == '' or band2_file.filename == '':
                    flash("请选择两个波段的图像！")
                    return redirect(request.url)
                    
                try:
                    # 保存上传的波段图像
                    band1_filename = secure_filename(band1_file.filename)
                    band2_filename = secure_filename(band2_file.filename)
                    
                    band1_path = os.path.join(app.config['UPLOAD_FOLDER'], band1_filename)
                    band2_path = os.path.join(app.config['UPLOAD_FOLDER'], band2_filename)
                    
                    band1_file.save(band1_path)
                    band2_file.save(band2_path)
                    
                    fusion_type = request.form.get('fusion_type', 'wavelet')
                    
                    fusion_processor = MultiBandFusion()
                    fused_path, colored_path = fusion_processor.fusion(
                        band1_path, band2_path, method=fusion_type)
                    
                    # 转换路径为URL
                    colored_url = url_for('static', 
                                        filename=os.path.relpath(colored_path, 'static').replace('\\', '/'))
                    
                    return render_template('index.html',
                                        uploaded_file=band1_path,
                                        processed_image=colored_url,  # 直接使用彩色图像作为处理结果
                                        operation='multiband_fusion')
                                
                except Exception as e:
                    print(f"融合处理错误: {str(e)}")
                    flash(f"图像融合失败：{str(e)}")
                    return redirect(request.url)
            elif operation == 'superglue_match':
                try:
                    print("\n=== SuperGlue特征匹配处理开始 ===")
                    print(f"请求方法: {request.method}")
                    print(f"Content-Type: {request.content_type}")
                    print(f"请求中的所有文件: {list(request.files.keys())}")
                    print(f"请求中的所有表单数据: {list(request.form.keys())}")
                    
                    if 'second_image' not in request.files:
                        print("错误：请求中没有second_image字段")
                        flash("请选择第二张图片进行匹配！")
                        return redirect(request.url)
                        
                    second_file = request.files['second_image']
                    print(f"第二张图片信息: filename={second_file.filename}, content_type={second_file.content_type}")
                    
                    if not second_file or second_file.filename == '':
                        print("错误：没有选择第二张图片")
                        flash("请选择第二张图片！")
                        return redirect(request.url)
                        
                    # 验证第一张图片
                    if not uploaded_file:
                        print("错误：没有上传第一张图片")
                        flash("请先上传第一张图片！")
                        return redirect(request.url)
                        
                    # 保存第二张图片
                    second_filename = secure_filename(second_file.filename)
                    second_filepath = os.path.join(app.config['UPLOAD_FOLDER'], second_filename)
                    second_file.save(second_filepath)
                    print(f"第二张图片已保存到: {second_filepath}")
                    
                    try:
                        print(f"开始匹配处理:\n图片1: {uploaded_file}\n图片2: {second_filepath}")
                        # 进行 SuperGlue 特征匹配
                        processed_image = superglue_matcher.match_images(
                            uploaded_file, second_filepath)
                        
                        print(f"匹配结果: {processed_image}")
                        
                        # 转换路径为URL
                        processed_image = os.path.relpath(processed_image, 'static')
                        processed_image = processed_image.replace("\\", "/")
                        processed_image = url_for('static', filename=processed_image)
                        
                        print(f"最终URL: {processed_image}")
                        
                        return render_template('index.html',
                                            processed_image=processed_image,
                                            uploaded_file=uploaded_file,
                                            operation='superglue_match')
                            
                    except Exception as e:
                        print(f"匹配处理错误: {str(e)}")
                        flash(f"特征匹配失败：{str(e)}")
                        return redirect(request.url)
                        
                except Exception as e:
                    print(f"处理请求错误: {str(e)}")
                    flash(f"处理失败：{str(e)}")
                    return redirect(request.url)
            elif operation == 'dehaze':
                processed_image = dark_channel_dehazing(uploaded_file)
            elif operation == 'cyclegan_dehaze':
                try:
                    print("开始CycleGAN去雾处理...")
                    session['processing_progress'] = 0
                    session['processing_stage'] = "准备CycleGAN模型..."
                    
                    def progress_callback(stage, progress):
                        session['processing_stage'] = stage
                        session['processing_progress'] = progress
                        print(f"处理进度: {stage} - {progress}%")
                        
                    # 检查模型文件是否存在
                    model_path = 'models/cyclegan_dehaze.pth'
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"模型文件不存在: {model_path}。请先下载模型文件。")
                        
                    processed_image = cyclegan_dehaze(uploaded_file, progress_callback)
                    
                    if not processed_image or not os.path.exists(processed_image):
                        raise Exception("处理后的图像文件不存在")
                        
                    print(f"处理完成，输出文件: {processed_image}")
                    
                except FileNotFoundError as e:
                    print(f"错误: {str(e)}")
                    flash(f"CycleGAN去雾处理失败：找不到模型文件，请先下载模型。")
                    return redirect(request.url)
                except Exception as e:
                    print(f"错误: {str(e)}")
                    flash(f"CycleGAN去雾处理失败：{str(e)}")
                    return redirect(request.url)
            elif operation == 'deblur':
                deblur_method = request.form.get('deblur_method', 'wiener')  # 获取去模糊方法
                if deblur_method == 'deblurgan':
                    processed_image = deblurgan_deblur(uploaded_file)
                else:
                    processed_image = deblur_image(uploaded_file, method=deblur_method)
                if processed_image is None:
                    flash("图像去模糊处理失败！")
                    return redirect(request.url)
            elif operation == 'restore':
                restore_method = request.form.get('restore_method', 'inpaint')
                if restore_method == 'inpaint':
                    processed_image = inpaint_photo_restoration(uploaded_file)
                elif restore_method == 'deep_learning':
                    processed_image = restore_old_photo(uploaded_file)
                elif restore_method == 'histogram_matching':
                    # 获取参考图片
                    reference_image = request.files.get('reference_image')
                    if not reference_image:
                        return '直方图匹配需要上传参考图片', 400
                    
                    # 保存参考图片
                    reference_filename = secure_filename(reference_image.filename)
                    reference_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_filename)
                    reference_image.save(reference_path)
                    
                    # 处理图片
                    processed_image = color_correction_with_histogram_matching(uploaded_file, reference_path)
                    
                    # 清理参考图片
                    os.remove(reference_path)

            else:
                flash("无效的操作类型，请重新选择！")
                return redirect(request.url)

            # 转换为相对路径以供模板使用
            processed_image = os.path.relpath(processed_image, 'static')
            processed_image = processed_image.replace("\\", "/")  # 替换反斜杠为正斜杠
            processed_image = url_for('static', filename=processed_image)

        except Exception as e:
            flash(f"处理图像时出错：{str(e)}")
            return redirect(request.url)

    return render_template('index.html', processed_image=processed_image, uploaded_file=uploaded_file)

@app.route('/crop_save', methods=['POST'])
def crop_save():
    """保存裁剪后的图像"""
    if 'cropped_image' not in request.files:
        print("未找到裁剪图像")
        return jsonify({'error': '未找到裁剪图像'}), 400

    cropped_image = request.files['cropped_image']
    uploaded_file = session.get('uploaded_file')
    if not uploaded_file:
        print("未找到原始图像")
        return jsonify({'error': '未找到原始图像'}), 400

    filename = os.path.basename(uploaded_file)
    name, ext = os.path.splitext(filename)
    cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_cropped{ext}")
    cropped_image.save(cropped_path)

    print(f"裁剪图像已保存到：{cropped_path}")
    flash("裁剪图像已保存！")
    return jsonify({'message': '裁剪图像已保存！'}), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        # 修复路径拼接问题
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename.replace('/', os.sep).replace('\\', os.sep))
        print(f"Serving file: {file_path}")
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving file: {str(e)}")
        flash(f"无法加载文件：{str(e)}")
        return redirect(url_for('index'))
    
    
@app.route('/reset', methods=['POST'])
def reset_file():
    """清除会话中的上传文件路径"""
    session.pop('uploaded_file', None)
    flash("已清除当前文件，请重新选择文件！")
    return redirect(url_for('index'))

# 添加进度查询接口
@app.route('/progress')
def get_progress():
    return jsonify({
        'stage': session.get('processing_stage', ''),
        'progress': session.get('processing_progress', 0)
    })

if __name__ == '__main__':
    app.run(debug=True)
    app.run(debug=True)