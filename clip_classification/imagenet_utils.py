"""
ImageNet数据集工具类
简化版 - 用于在项目中快速使用
"""

import os
import scipy.io
from typing import Dict, Optional


class ImageNetMetadata:
    """ImageNet元数据管理类"""
    
    def __init__(self, meta_file: str):
        """
        初始化ImageNet元数据
        
        Args:
            meta_file: meta.mat文件路径
        """
        self.meta_file = meta_file
        self.wnid_to_info = self._parse_metadata()
    
    def _parse_metadata(self) -> Dict[str, Dict]:
        """解析meta.mat文件"""
        meta = scipy.io.loadmat(self.meta_file, squeeze_me=True, struct_as_record=False)
        synsets = meta['synsets']
        
        wnid_to_info = {}
        for synset in synsets:
            wnid = synset.WNID
            wnid_to_info[wnid] = {
                'wnid': wnid,
                'ilsvrc2012_id': synset.ILSVRC2012_ID,
                'words': synset.words,
                'gloss': synset.gloss if hasattr(synset, 'gloss') else "",
            }
        
        return wnid_to_info
    
    def get_class_name(self, wnid: str) -> str:
        """
        根据WNID获取类别名称
        
        Args:
            wnid: WordNet ID (如 n01530575)
        
        Returns:
            类别名称
        """
        if wnid in self.wnid_to_info:
            return self.wnid_to_info[wnid]['words']
        return f"Unknown ({wnid})"
    
    def get_class_description(self, wnid: str) -> str:
        """
        获取类别的详细描述
        
        Args:
            wnid: WordNet ID
        
        Returns:
            类别描述
        """
        if wnid in self.wnid_to_info:
            return self.wnid_to_info[wnid]['gloss']
        return ""
    
    def get_wnid_from_path(self, image_path: str) -> Optional[str]:
        """
        从图像路径中提取WNID
        
        Args:
            image_path: 图像路径
        
        Returns:
            WNID或None
        """
        # 从路径中查找
        parts = image_path.replace('\\', '/').split('/')
        for part in reversed(parts):
            if part.startswith('n') and len(part) == 9:
                return part
        
        # 从文件名中查找
        filename = os.path.basename(image_path)
        if filename.startswith('n') and '_' in filename:
            return filename.split('_')[0]
        
        return None
    
    def get_name_from_path(self, image_path: str) -> str:
        """
        从图像路径直接获取类别名称
        
        Args:
            image_path: 图像路径
        
        Returns:
            类别名称
        """
        wnid = self.get_wnid_from_path(image_path)
        if wnid:
            return self.get_class_name(wnid)
        return "Unknown"
    
    def __len__(self):
        """返回类别总数"""
        return len(self.wnid_to_info)
    
    def __contains__(self, wnid):
        """检查WNID是否存在"""
        return wnid in self.wnid_to_info


# 便捷函数
def create_imagenet_metadata(
    data_root: str = r"G:\Thomas\3_1_project\data\ImageNet-data"
) -> ImageNetMetadata:
    """
    创建ImageNet元数据对象的便捷函数
    
    Args:
        data_root: ImageNet数据根目录
    
    Returns:
        ImageNetMetadata对象
    """
    meta_file = os.path.join(
        data_root, 
        "meta", 
        "ILSVRC2012_devkit_t12", 
        "data", 
        "meta.mat"
    )
    
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"找不到元数据文件: {meta_file}")
    
    return ImageNetMetadata(meta_file)


if __name__ == '__main__':
    # 使用示例
    print("创建ImageNet元数据对象...")
    metadata = create_imagenet_metadata()
    print(f"✓ 加载了 {len(metadata)} 个类别\n")
    
    # 示例1: 通过WNID查询类别名称
    wnid = "n01530575"
    print(f"示例1: WNID查询")
    print(f"  WNID: {wnid}")
    print(f"  名称: {metadata.get_class_name(wnid)}")
    print(f"  描述: {metadata.get_class_description(wnid)}\n")
    
    # 示例2: 从图像路径获取类别名称
    image_path = r"train\n01530575\n01530575_10007.JPEG"
    print(f"示例2: 从路径获取类别")
    print(f"  路径: {image_path}")
    print(f"  类别: {metadata.get_name_from_path(image_path)}\n")
    
    # 示例3: 批量处理
    print("示例3: 批量处理几个WNID")
    sample_wnids = ["n01530575", "n02099267", "n03594734"]
    for wnid in sample_wnids:
        if wnid in metadata:
            print(f"  {wnid}: {metadata.get_class_name(wnid)}")
